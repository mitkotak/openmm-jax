#ifndef OPENMM_PJRT_RUNTIME_H_
#define OPENMM_PJRT_RUNTIME_H_

#include "OpenMmPjrtOutputLifetime.h"
#include "PjrtClientSession.h"
#include "PjrtHandles.h"
#include <cuda.h>
#include <functional>
#include <string>

namespace JaxPlugin {


struct OpenMmPjrtInputs {
    CUdeviceptr positions = 0;       
    CUdeviceptr boxVectors = 0;
    int numParticles = 0;
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

struct OpenMmPjrtExecutionResult {
    double energy = 0.0;
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
    void waitOnStream(CUstream stream, CUevent readyEvent);

    PjrtClientSession session;
    PjrtLoadedExecutablePtr forceExecutable;
    PjrtLoadedExecutablePtr energyExecutable;
    PjrtLoadedExecutablePtr energyAndForcesExecutable;
    OpenMmPjrtOutputLifetime outputLifetime;
    bool stablehloUsesDoublePrecisionReal = false;
};

} // namespace JaxPlugin

#endif
