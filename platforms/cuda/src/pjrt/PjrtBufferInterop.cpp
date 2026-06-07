#include "PjrtBufferInterop.h"
#include <stdexcept>

using namespace JaxPlugin;
using namespace std;

// No-op callback. PJRT buffer views wrap existing OpenMM device memory and
// must not free it when the view is destroyed.
static void pjrtViewDeleteCallback(void*, void*) {
}

PjrtBufferPtr JaxPlugin::createViewOfDeviceBuffer(PjrtClientSession& session,
        CUdeviceptr devicePointer, const int64_t* dims, size_t numDims,
        PJRT_Buffer_Type elementType, CUstream stream, int deviceIndex,
        const string& label) {
    if (!session.hasDevice(deviceIndex))
        throw runtime_error("JaxForce PJRT: device index out of range for " + label);

    int64_t minorToMajor[3];
    for (size_t i = 0; i < numDims; i++)
        minorToMajor[i] = static_cast<int64_t>(numDims - 1 - i);

    PJRT_Buffer_MemoryLayout layout;
    layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
    layout.extension_start = nullptr;
    layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
    layout.tiled.extension_start = nullptr;
    layout.tiled.minor_to_major = minorToMajor;
    layout.tiled.minor_to_major_size = numDims;
    layout.tiled.tile_dims = nullptr;
    layout.tiled.tile_dim_sizes = nullptr;
    layout.tiled.num_tiles = 0;
    layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;

    const PJRT_Api* api = session.api();
    PJRT_Client_CreateViewOfDeviceBuffer_Args viewArgs;
    viewArgs.struct_size = PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
    viewArgs.extension_start = nullptr;
    viewArgs.client = session.client();
    viewArgs.device_buffer_ptr = reinterpret_cast<void*>(devicePointer);
    viewArgs.dims = dims;
    viewArgs.num_dims = numDims;
    viewArgs.element_type = elementType;
    viewArgs.layout = &layout;
    viewArgs.device = session.device(deviceIndex);
    viewArgs.on_delete_callback = pjrtViewDeleteCallback;
    viewArgs.on_delete_callback_arg = nullptr;
    viewArgs.stream = reinterpret_cast<intptr_t>(stream);
    viewArgs.buffer = nullptr;
    viewArgs.memory = nullptr;

    session.pluginLibrary().checkError(api->PJRT_Client_CreateViewOfDeviceBuffer(&viewArgs),
            "PJRT_Client_CreateViewOfDeviceBuffer(" + label + ")");
    if (viewArgs.buffer == nullptr)
        throw runtime_error(
                "JaxForce PJRT: CreateViewOfDeviceBuffer returned null for " + label);

    return PjrtBufferPtr(viewArgs.buffer, makeBufferDeleter(api));
}

CUdeviceptr JaxPlugin::getOpaqueDeviceMemoryDataPointer(PjrtClientSession& session,
        PjrtBufferPtr& buffer, const string& label) {
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args pointerArgs;
    pointerArgs.struct_size = PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE;
    pointerArgs.extension_start = nullptr;
    pointerArgs.buffer = buffer.get();
    pointerArgs.device_memory_ptr = nullptr;

    session.pluginLibrary().checkError(
            session.api()->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&pointerArgs),
            "PJRT_Buffer_OpaqueDeviceMemoryDataPointer(" + label + ")");
    if (pointerArgs.device_memory_ptr == nullptr)
        throw runtime_error("JaxForce PJRT: output buffer has null device pointer for " + label);

    return reinterpret_cast<CUdeviceptr>(pointerArgs.device_memory_ptr);
}
