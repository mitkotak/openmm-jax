#ifndef OPENMM_PJRT_BUFFER_INTEROP_H_
#define OPENMM_PJRT_BUFFER_INTEROP_H_

#include "PjrtClientSession.h"
#include "PjrtHandles.h"
#include <cuda.h>
#include <string>

namespace JaxPlugin {

PjrtBufferPtr createViewOfDeviceBuffer(PjrtClientSession& session,
        CUdeviceptr devicePointer, const int64_t* dims, size_t numDims,
        PJRT_Buffer_Type elementType, CUstream stream, int deviceIndex,
        const std::string& label);

CUdeviceptr getOpaqueDeviceMemoryDataPointer(PjrtClientSession& session,
        PjrtBufferPtr& buffer, const std::string& label);

} // namespace JaxPlugin

#endif
