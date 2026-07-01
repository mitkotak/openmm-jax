#ifndef OPENMM_PJRT_EXECUTION_TYPES_H_
#define OPENMM_PJRT_EXECUTION_TYPES_H_

#include "PjrtHandles.h"
#include <array>
#include <stdexcept>
#include <utility>

namespace JaxPlugin {

enum class RequestedOutputs {
    Forces,
    Energy,
    EnergyAndForces,
};

struct SelectedPjrtProgram {
    PJRT_LoadedExecutable* executable = nullptr;
    size_t outputCount = 0;
    int forceOutputIndex = -1;
    int energyOutputIndex = -1;
    const char* label = nullptr;
};

struct PjrtInputBuffers {
    std::array<PjrtBufferPtr, 2> buffers;
    size_t count = 0;

    void push(PjrtBufferPtr buffer) {
        if (count >= buffers.size())
            throw std::runtime_error("JaxForce PJRT: too many input buffers");
        buffers[count++] = std::move(buffer);
    }
};

struct PjrtOutputBuffers {
    std::array<PjrtBufferPtr, 2> buffers;
    PjrtEventPtr completeEvent;
};

} // namespace JaxPlugin

#endif
