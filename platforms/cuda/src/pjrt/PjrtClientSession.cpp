#include "PjrtClientSession.h"
#include "PjrtLoadedExecutable.h"
#include "pjrt_cuda_stream_extension.h"
#include <array>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <utility>

using namespace JaxPlugin;
using namespace std;

static PJRT_NamedValue pjrtOption(const char* name, PJRT_NamedValue_Type type) {
    PJRT_NamedValue option;
    option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
    option.extension_start = nullptr;
    option.name = name;
    option.name_size = strlen(name);
    option.type = type;
    option.value_size = 1;
    return option;
}

PjrtClientSession::PjrtClientSession(PjrtClientSession&& other) noexcept :
        library(std::move(other.library)),
        clientHandle(std::move(other.clientHandle)),
        devices(std::move(other.devices)),
        streamExtension(other.streamExtension) {
    other.streamExtension = nullptr;
}

PjrtClientSession& PjrtClientSession::operator=(PjrtClientSession&& other) {
    if (this != &other) {
        close();
        library = std::move(other.library);
        clientHandle = std::move(other.clientHandle);
        devices = std::move(other.devices);
        streamExtension = other.streamExtension;
        other.streamExtension = nullptr;
    }
    return *this;
}

PjrtClientSession::~PjrtClientSession() {
    try { close(); } catch (...) {}
}

void PjrtClientSession::initialize(const string& pluginPath) {
    close();
    if (pluginPath.empty())
        throw runtime_error("JaxForce PJRT: PJRT plugin path must be provided");

    PjrtPluginLibrary newLibrary;
    newLibrary.open(pluginPath);

    const PJRT_Api* api = newLibrary.getApi();
    validateRequiredPjrtApi(api);

    PJRT_Client_Create_Args createArgs;
    createArgs.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    createArgs.extension_start = nullptr;
    array<PJRT_NamedValue, 2> createOptions;
    size_t numCreateOptions = 0;
    PJRT_NamedValue& allocatorOption = createOptions[numCreateOptions++];
    allocatorOption = pjrtOption("allocator", PJRT_NamedValue_kString);
    allocatorOption.string_value = "default";
    allocatorOption.value_size = strlen(allocatorOption.string_value);

    PJRT_NamedValue& preallocateOption = createOptions[numCreateOptions++];
    preallocateOption = pjrtOption("preallocate", PJRT_NamedValue_kBool);
    preallocateOption.bool_value = false;

    createArgs.create_options = createOptions.data();
    createArgs.num_options = numCreateOptions;
    createArgs.kv_get_callback = nullptr;
    createArgs.kv_get_user_arg = nullptr;
    createArgs.kv_put_callback = nullptr;
    createArgs.kv_put_user_arg = nullptr;
    createArgs.client = nullptr;
    createArgs.kv_try_get_callback = nullptr;
    createArgs.kv_try_get_user_arg = nullptr;

    newLibrary.checkError(api->PJRT_Client_Create(&createArgs), "PJRT_Client_Create");
    PjrtClientPtr newClient(createArgs.client, makeClientDeleter(api));
    if (newClient == nullptr)
        throw runtime_error("JaxForce PJRT: PJRT_Client_Create returned null client");

    PJRT_Client_Devices_Args deviceArgs;
    deviceArgs.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
    deviceArgs.extension_start = nullptr;
    deviceArgs.client = newClient.get();
    deviceArgs.devices = nullptr;
    deviceArgs.num_devices = 0;

    newLibrary.checkError(api->PJRT_Client_Devices(&deviceArgs), "PJRT_Client_Devices");
    vector<PJRT_Device*> newDevices(deviceArgs.devices, deviceArgs.devices + deviceArgs.num_devices);
    PJRT_Stream_Extension* newStreamExtension =
            newLibrary.findExtension<PJRT_Stream_Extension>(PJRT_Extension_Type_Stream);

    library = std::move(newLibrary);
    clientHandle = std::move(newClient);
    devices = std::move(newDevices);
    streamExtension = newStreamExtension;
}

void PjrtClientSession::close() {
    exception_ptr firstError;
    try {
        clientHandle.reset();
    }
    catch (...) {
        if (!firstError)
            firstError = current_exception();
    }
    clearClientState();
    library.close();
    if (firstError)
        rethrow_exception(firstError);
}

const PJRT_Api* PjrtClientSession::api() const {
    return library.getApi();
}

PjrtPluginLibrary& PjrtClientSession::pluginLibrary() {
    return library;
}

const PjrtPluginLibrary& PjrtClientSession::pluginLibrary() const {
    return library;
}

PJRT_Client* PjrtClientSession::client() const {
    return clientHandle.get();
}

PJRT_Device* PjrtClientSession::device(int deviceIndex) const {
    if (!hasDevice(deviceIndex))
        throw runtime_error("JaxForce PJRT: device index out of range");
    return devices[static_cast<size_t>(deviceIndex)];
}

bool PjrtClientSession::hasDevice(int deviceIndex) const {
    return deviceIndex >= 0 && static_cast<size_t>(deviceIndex) < devices.size();
}

CUstream PjrtClientSession::getStreamForExternalReadyEvents(int deviceIndex) {
    if (streamExtension == nullptr || streamExtension->get_stream == nullptr)
        throw runtime_error(
                "JaxForce PJRT: CUDA PJRT stream extension is required for OpenMM stream handoff");

    PJRT_Get_Stream_For_External_Ready_Events_Args streamArgs;
    streamArgs.struct_size = PJRT_Get_Stream_For_External_Ready_Events_Args_STRUCT_SIZE;
    streamArgs.device = device(deviceIndex);
    streamArgs.stream = 0;
    library.checkError(streamExtension->get_stream(&streamArgs),
            "PJRT_Get_Stream_For_External_Ready_Events");
    return reinterpret_cast<CUstream>(streamArgs.stream);
}

void PjrtClientSession::clearClientState() {
    clientHandle.reset();
    devices.clear();
    streamExtension = nullptr;
}
