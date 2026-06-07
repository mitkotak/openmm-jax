#include "PjrtLoadedExecutable.h"
#include <array>
#include <stdexcept>

using namespace JaxPlugin;
using namespace std;

void JaxPlugin::validateRequiredPjrtApi(const PJRT_Api* api) {
    if (api == nullptr ||
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

PjrtEventPtr JaxPlugin::executeLoadedExecutable(PjrtClientSession& session,
        PJRT_LoadedExecutable* executable, PjrtBufferPtr* inputs,
        size_t numInputs, int deviceIndex, PjrtBufferPtr* outputs,
        size_t numOutputs, const string& label, const string& callLocation) {
    const PJRT_Api* api = session.api();

    std::array<PJRT_Buffer*, 2> inputPtrs = {nullptr, nullptr};
    for (size_t i = 0; i < numInputs; i++)
        inputPtrs[i] = inputs[i].get();

    PJRT_Buffer* const* argumentLists[1] = {inputPtrs.data()};
    std::array<PJRT_Buffer*, 2> outputList = {nullptr, nullptr};
    PJRT_Buffer** outputLists[1] = {outputList.data()};
    std::array<PJRT_Event*, 1> events = {nullptr};

    int64_t nonDonatable[2] = {0, 0};
    for (size_t i = 0; i < numInputs; i++)
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
    options.num_non_donatable_input_indices = numInputs;
    options.context = nullptr;
    options.call_location = callLocation.c_str();
    options.num_tasks = 0;
    options.task_ids = nullptr;
    options.incarnation_ids = nullptr;
    options.multi_slice_config = nullptr;

    PJRT_LoadedExecutable_Execute_Args executeArgs;
    executeArgs.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    executeArgs.extension_start = nullptr;
    executeArgs.executable = executable;
    executeArgs.options = &options;
    executeArgs.argument_lists = argumentLists;
    executeArgs.num_devices = 1;
    executeArgs.num_args = numInputs;
    executeArgs.output_lists = outputLists;
    executeArgs.device_complete_events = events.data();
    executeArgs.execute_device = session.device(deviceIndex);

    PjrtErrorPtr executeError(api->PJRT_LoadedExecutable_Execute(&executeArgs),
            makeErrorDeleter(api));
    session.pluginLibrary().checkError(executeError.release(),
            "PJRT_LoadedExecutable_Execute(" + label + ")");

    for (size_t i = 0; i < numOutputs; i++) {
        if (outputList[i] != nullptr)
            outputs[i] = PjrtBufferPtr(outputList[i], makeBufferDeleter(api));
    }
    for (size_t i = 0; i < numOutputs; i++)
        if (outputs[i] == nullptr)
            throw runtime_error("JaxForce PJRT: execute returned null output for " + label);

    return PjrtEventPtr(events[0], makeEventDeleter(api));
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
