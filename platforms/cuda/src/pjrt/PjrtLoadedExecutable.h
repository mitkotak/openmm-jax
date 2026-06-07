#ifndef OPENMM_PJRT_LOADED_EXECUTABLE_H_
#define OPENMM_PJRT_LOADED_EXECUTABLE_H_

#include "PjrtClientSession.h"
#include "PjrtHandles.h"
#include <string>

namespace JaxPlugin {

void validateRequiredPjrtApi(const PJRT_Api* api);

PjrtLoadedExecutablePtr compileStablehloExecutable(PjrtClientSession& session,
        const std::string& mlir, const std::string& compileOptions,
        const std::string& label);

PjrtEventPtr executeLoadedExecutable(PjrtClientSession& session,
        PJRT_LoadedExecutable* executable, PjrtBufferPtr* inputs,
        size_t numInputs, int deviceIndex, PjrtBufferPtr* outputs,
        size_t numOutputs, const std::string& label,
        const std::string& callLocation);

void awaitDeviceCompleteEvent(PjrtClientSession& session, PjrtEventPtr& event,
        const std::string& label);

} // namespace JaxPlugin

#endif
