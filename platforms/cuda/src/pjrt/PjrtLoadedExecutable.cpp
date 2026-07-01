#include "PjrtLoadedExecutable.h"
#include <array>
#include <stdexcept>
#include <sstream>

using namespace JaxPlugin;
using namespace std;

void JaxPlugin::validateRequiredPjrtApi(const PJRT_Api* api) {
    if (api == nullptr)
        throw runtime_error("JaxForce PJRT: plugin API is missing required compile/execute entry points");
    size_t requiredSize = PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Client_CreateViewOfDeviceBuffer);
    if (api->struct_size < requiredSize) {
        stringstream message;
        message << "JaxForce PJRT: plugin API table is too old for OpenMM-JAX "
                << "CUDA interop (plugin struct_size=" << api->struct_size
                << ", required at least " << requiredSize << ")";
        throw runtime_error(message.str());
    }
    if (
            api->PJRT_Client_Compile == nullptr ||
            api->PJRT_Client_Destroy == nullptr ||
            api->PJRT_Client_CreateViewOfDeviceBuffer == nullptr ||
            api->PJRT_LoadedExecutable_Execute == nullptr ||
            api->PJRT_LoadedExecutable_Destroy == nullptr ||
            api->PJRT_Event_Await == nullptr ||
            api->PJRT_Event_Destroy == nullptr ||
            api->PJRT_Buffer_OpaqueDeviceMemoryDataPointer == nullptr ||
            api->PJRT_Buffer_Destroy == nullptr)
        throw runtime_error("JaxForce PJRT: plugin API is missing required compile/execute entry points");
}

PjrtLoadedExecutablePtr JaxPlugin::compileStablehloExecutable(PjrtClientSession& session,
        const string& mlir, const string& compileOptions,
        const string& label) {
    if (compileOptions.empty())
        throw runtime_error("JaxForce PJRT: compile options must be provided before compiling "+label);

    const PJRT_Api* api = session.api();
    validateRequiredPjrtApi(api);
    string format = "mlir";

    PJRT_Program program;
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    program.extension_start = nullptr;
    program.code = const_cast<char*>(mlir.data());
    program.code_size = mlir.size();
    program.format = const_cast<char*>(format.data());
    program.format_size = format.size();

    PJRT_Client_Compile_Args compileArgs;
    compileArgs.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    compileArgs.extension_start = nullptr;
    compileArgs.client = session.client();
    compileArgs.program = &program;
    compileArgs.compile_options = compileOptions.data();
    compileArgs.compile_options_size = compileOptions.size();
    compileArgs.executable = nullptr;

    session.pluginLibrary().checkError(api->PJRT_Client_Compile(&compileArgs),
            "PJRT_Client_Compile("+label+")");
    if (compileArgs.executable == nullptr)
        throw runtime_error("JaxForce PJRT: PJRT_Client_Compile returned null "+label+" executable");

    return PjrtLoadedExecutablePtr(compileArgs.executable, makeLoadedExecutableDeleter(api));
}

PjrtOutputBuffers JaxPlugin::executeLoadedExecutable(PjrtClientSession& session,
        const SelectedPjrtProgram& program, PjrtInputBuffers& inputs,
        int deviceIndex) {
    const PJRT_Api* api = session.api();
    if (program.executable == nullptr)
        throw runtime_error("JaxForce PJRT: selected program has no executable");
    if (inputs.count > inputs.buffers.size())
        throw runtime_error("JaxForce PJRT: too many input buffers");
    if (program.outputCount > 2)
        throw runtime_error("JaxForce PJRT: too many output buffers for " + string(program.label));

    std::array<PJRT_Buffer*, 2> inputPtrs = {nullptr, nullptr};
    for (size_t i = 0; i < inputs.count; i++)
        inputPtrs[i] = inputs.buffers[i].get();

    PJRT_Buffer* const* argumentLists[1] = {inputPtrs.data()};
    std::array<PJRT_Buffer*, 2> outputList = {nullptr, nullptr};
    PJRT_Buffer** outputLists[1] = {outputList.data()};
    std::array<PJRT_Event*, 1> events = {nullptr};

    int64_t nonDonatable[2] = {0, 0};
    for (size_t i = 0; i < inputs.count; i++)
        nonDonatable[i] = static_cast<int64_t>(i);

    PJRT_ExecuteOptions options;
    options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    options.extension_start = nullptr;
    options.send_callbacks = nullptr;
    options.recv_callbacks = nullptr;
    options.num_send_ops = 0;
    options.num_recv_ops = 0;
    options.launch_id = 0;
    options.non_donatable_input_indices = nonDonatable;
    options.num_non_donatable_input_indices = inputs.count;
    options.context = nullptr;
    options.call_location = program.label;
    options.num_tasks = 0;
    options.task_ids = nullptr;
    options.incarnation_ids = nullptr;
    options.multi_slice_config = nullptr;

    PJRT_LoadedExecutable_Execute_Args executeArgs;
    executeArgs.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    executeArgs.extension_start = nullptr;
    executeArgs.executable = program.executable;
    executeArgs.options = &options;
    executeArgs.argument_lists = argumentLists;
    executeArgs.num_devices = 1;
    executeArgs.num_args = inputs.count;
    executeArgs.output_lists = outputLists;
    executeArgs.device_complete_events = events.data();
    executeArgs.execute_device = session.device(deviceIndex);

    PjrtErrorPtr executeError(api->PJRT_LoadedExecutable_Execute(&executeArgs),
            makeErrorDeleter(api));
    session.pluginLibrary().checkError(executeError.release(),
            "PJRT_LoadedExecutable_Execute(" + string(program.label) + ")");

    PjrtOutputBuffers result;
    for (size_t i = 0; i < program.outputCount; i++) {
        if (outputList[i] != nullptr)
            result.buffers[i] = PjrtBufferPtr(outputList[i], makeBufferDeleter(api));
    }
    for (size_t i = 0; i < program.outputCount; i++)
        if (result.buffers[i] == nullptr)
            throw runtime_error("JaxForce PJRT: execute returned null output for " + string(program.label));

    result.completeEvent = PjrtEventPtr(events[0], makeEventDeleter(api));
    return result;
}

void JaxPlugin::awaitDeviceCompleteEvent(PjrtClientSession& session,
        PjrtEventPtr& event, const string& label) {
    if (event != nullptr) {
        PJRT_Event_Await_Args awaitArgs;
        awaitArgs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        awaitArgs.extension_start = nullptr;
        awaitArgs.event = event.get();
        session.pluginLibrary().checkError(session.api()->PJRT_Event_Await(&awaitArgs),
                "PJRT_Event_Await(" + label + ")");
    }
}
