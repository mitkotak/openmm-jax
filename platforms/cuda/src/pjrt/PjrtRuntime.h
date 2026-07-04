#ifndef OPENMM_PJRT_RUNTIME_H_
#define OPENMM_PJRT_RUNTIME_H_

#include "OpenMmPjrtOutputLifetime.h"
#include "PjrtClientSession.h"
#include "PjrtExecutionTypes.h"
#include "PjrtHandles.h"
#include <cuda.h>
#include <functional>
#include <string>

namespace JaxPlugin {


struct OpenMmPjrtInputs {
    CUdeviceptr positions = 0;
    CUdeviceptr boxVectors = 0;
    int numInputParticles = 0;
    int deviceIndex = 0;
    CUstream stream = nullptr;
    bool usePeriodic = false;
    CUevent inputReadyEvent = nullptr;
    bool useDoublePrecisionReal = false;
};

class OpenMmPjrtForceOutput {
public:
    OpenMmPjrtForceOutput() = default;
    OpenMmPjrtForceOutput(const OpenMmPjrtForceOutput&) = delete;
    OpenMmPjrtForceOutput& operator=(const OpenMmPjrtForceOutput&) = delete;
    OpenMmPjrtForceOutput(OpenMmPjrtForceOutput&&) noexcept = default;
    OpenMmPjrtForceOutput& operator=(OpenMmPjrtForceOutput&&) noexcept = default;

    void consumeOnStream(CUstream openmmStream,
            const std::function<void(CUdeviceptr)>& consumer);

private:
    friend class PjrtRuntime;

    OpenMmPjrtForceOutput(OpenMmPjrtOutputLifetime* lifetime, PjrtBufferPtr buffer,
            CUdeviceptr pointer);

    OpenMmPjrtOutputLifetime* lifetime = nullptr;
    PjrtBufferPtr buffer;
    CUdeviceptr pointer = 0;
};

class OpenMmPjrtEnergyOutput {
public:
    OpenMmPjrtEnergyOutput() = default;
    OpenMmPjrtEnergyOutput(const OpenMmPjrtEnergyOutput&) = delete;
    OpenMmPjrtEnergyOutput& operator=(const OpenMmPjrtEnergyOutput&) = delete;
    OpenMmPjrtEnergyOutput(OpenMmPjrtEnergyOutput&&) noexcept = default;
    OpenMmPjrtEnergyOutput& operator=(OpenMmPjrtEnergyOutput&&) noexcept = default;

    double copyScalarToHostDouble(CUstream stream);
    void destroy();
    void release() noexcept;

private:
    friend class PjrtRuntime;

    OpenMmPjrtEnergyOutput(PjrtBufferPtr buffer, CUdeviceptr pointer,
            PJRT_Buffer_Type type);

    PjrtBufferPtr buffer;
    CUdeviceptr pointer = 0;
    PJRT_Buffer_Type type = PJRT_Buffer_Type_INVALID;
};

struct OpenMmPjrtExecutionResult {
    OpenMmPjrtEnergyOutput energyOutput;
    OpenMmPjrtForceOutput forceOutput;
};

class PjrtRuntime {
public:
    ~PjrtRuntime();

    void initialize(const std::string& pluginPath, const std::string& forceMlir,
            const std::string& energyMlir,
            const std::string& energyAndForcesMlir,
            const std::string& compileOptions);
    void close();

    OpenMmPjrtExecutionResult execute(const OpenMmPjrtInputs& inputs,
            bool includeForces, bool includeEnergy);

private:
    SelectedPjrtProgram selectProgram(RequestedOutputs outputs) const;
    void validatePrecision(const OpenMmPjrtInputs& inputs) const;
    PjrtInputBuffers createInputViews(const OpenMmPjrtInputs& inputs,
            const SelectedPjrtProgram& program);
    OpenMmPjrtExecutionResult consumeOutputs(PjrtOutputBuffers outputs,
            const SelectedPjrtProgram& program);

    PjrtClientSession session;
    PjrtLoadedExecutablePtr forceExecutable;
    PjrtLoadedExecutablePtr energyExecutable;
    PjrtLoadedExecutablePtr energyAndForcesExecutable;
    OpenMmPjrtOutputLifetime outputLifetime;
    bool stablehloUsesDoublePrecisionReal = false;
};

} // namespace JaxPlugin

#endif
