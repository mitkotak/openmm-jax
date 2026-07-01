#ifndef OPENMM_PJRT_LOADED_EXECUTABLE_H_
#define OPENMM_PJRT_LOADED_EXECUTABLE_H_

#include "PjrtClientSession.h"
#include "PjrtExecutionTypes.h"
#include "PjrtHandles.h"
#include <string>

namespace JaxPlugin {

void validateRequiredPjrtApi(const PJRT_Api* api);

PjrtLoadedExecutablePtr compileStablehloExecutable(PjrtClientSession& session,
        const std::string& mlir, const std::string& compileOptions,
        const std::string& label);

PjrtOutputBuffers executeLoadedExecutable(PjrtClientSession& session,
        const SelectedPjrtProgram& program, PjrtInputBuffers& inputs,
        int deviceIndex);

void awaitDeviceCompleteEvent(PjrtClientSession& session, PjrtEventPtr& event,
        const std::string& label);

} // namespace JaxPlugin

#endif
