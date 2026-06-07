#include "PjrtRuntime.h"
#include "PjrtBufferInterop.h"
#include "PjrtLoadedExecutable.h"
#include <array>
#include <stdexcept>
#include <utility>

using namespace JaxPlugin;
using namespace std;

PjrtRuntime::~PjrtRuntime() {
    try { close(); } catch (...) {}
}


OpenMmPjrtForceOutput::OpenMmPjrtForceOutput(OpenMmPjrtOutputLifetime* lifetime,
        PjrtBufferPtr buffer, CUdeviceptr pointer) :
        lifetime(lifetime), buffer(std::move(buffer)), pointer(pointer) {
}

void OpenMmPjrtForceOutput::consumeOnStream(CUstream openmmStream,
        const function<void(CUdeviceptr)>& consumer) {
    if (lifetime == nullptr)
        throw runtime_error("JaxForce PJRT: force output is not available");
    lifetime->consumeForceOutput(std::move(buffer), pointer, openmmStream, consumer);
    lifetime = nullptr;
    pointer = 0;
}


void PjrtRuntime::close() {
    outputLifetime.reset();

    forceExecutable.reset();
    energyExecutable.reset();
    energyAndForcesExecutable.reset();
    session.close();
}

void PjrtRuntime::initialize(const string& pluginPath, const string& forceMlir,
        const string& energyMlir, const string& energyAndForcesMlir,
        const string& compileOptions) {
    close();

    if (forceMlir.empty() || energyMlir.empty() || energyAndForcesMlir.empty())
        throw runtime_error(
                "JaxForce PJRT: force, energy, and energy+forces StableHLO programs must all be provided");
    PjrtClientSession newSession;
    newSession.initialize(pluginPath);

    PjrtLoadedExecutablePtr newForceExecutable =
            compileStablehloExecutable(newSession, forceMlir, compileOptions, "force");
    PjrtLoadedExecutablePtr newEnergyExecutable =
            compileStablehloExecutable(newSession, energyMlir, compileOptions, "energy");
    PjrtLoadedExecutablePtr newEnergyAndForcesExecutable =
            compileStablehloExecutable(newSession, energyAndForcesMlir, compileOptions, "energy+forces");

    session = std::move(newSession);
    forceExecutable = std::move(newForceExecutable);
    energyExecutable = std::move(newEnergyExecutable);
    energyAndForcesExecutable = std::move(newEnergyAndForcesExecutable);
}


OpenMmPjrtExecutionResult PjrtRuntime::execute(
        const OpenMmPjrtInputs& pjrtInputs, bool includeForces,
        bool includeEnergy) {
    OpenMmPjrtExecutionResult result;
    if (!includeForces && !includeEnergy)
        return result;

    outputLifetime.cleanupBeforeExecution();

    struct ExecutableChoice {
        PJRT_LoadedExecutable* executable = nullptr;
        size_t numOutputs = 0;
        int forceOutputIndex = -1;
        int energyOutputIndex = -1;
        const char* label = nullptr;
        const char* callLocation = nullptr;
    } choice;
    if (includeForces && includeEnergy)
        choice = {energyAndForcesExecutable.get(), 2, 1, 0,
                "energy+forces", "OpenMM-JAX energy+forces"};
    else if (includeForces)
        choice = {forceExecutable.get(), 1, 0, -1,
                "force", "OpenMM-JAX force"};
    else
        choice = {energyExecutable.get(), 1, -1, 0,
                "energy", "OpenMM-JAX energy"};

    CUstream inputStream = session.getStreamForExternalReadyEvents(pjrtInputs.deviceIndex);
    if (pjrtInputs.inputReadyEvent != nullptr)
        waitOnStream(inputStream, pjrtInputs.inputReadyEvent);

    std::array<PjrtBufferPtr, 2> inputBuffers;
    size_t numInputs = 0;
    int64_t positionDims[2] = {pjrtInputs.numParticles, 3};
    inputBuffers[numInputs++] = createViewOfDeviceBuffer(session,
            pjrtInputs.positions, positionDims, 2,
            PJRT_Buffer_Type_F32, inputStream, pjrtInputs.deviceIndex,
            string(choice.label) + " positions");
    if (pjrtInputs.usePeriodic) {
        int64_t boxDims[2] = {3, 3};
        inputBuffers[numInputs++] = createViewOfDeviceBuffer(session,
                pjrtInputs.boxVectors, boxDims, 2,
                PJRT_Buffer_Type_F32, inputStream, pjrtInputs.deviceIndex,
                string(choice.label) + " boxVectors");
    }

    std::array<PjrtBufferPtr, 2> outputBuffers;
    PjrtEventPtr completeEvent = executeLoadedExecutable(session,
            choice.executable, inputBuffers.data(),
            numInputs, pjrtInputs.deviceIndex, outputBuffers.data(), choice.numOutputs,
            choice.label, choice.callLocation);
    awaitDeviceCompleteEvent(session, completeEvent, choice.label);

    if (choice.energyOutputIndex >= 0) {
        size_t index = static_cast<size_t>(choice.energyOutputIndex);
        CUdeviceptr energyPointer = getOpaqueDeviceMemoryDataPointer(session,
                outputBuffers[index], string(choice.label) + " energy");
        float energyFloat = 0.0f;
        CUresult copyResult = cuMemcpyDtoHAsync(
                &energyFloat, energyPointer, sizeof(float), pjrtInputs.stream);
        if (copyResult != CUDA_SUCCESS)
            throw runtime_error("JaxForce PJRT: failed to copy energy scalar from device");
        CUresult syncResult = cuStreamSynchronize(pjrtInputs.stream);
        if (syncResult != CUDA_SUCCESS)
            throw runtime_error("JaxForce PJRT: failed to synchronize energy scalar copy");
        result.energy = static_cast<double>(energyFloat);
        outputBuffers[index].reset();
    }

    if (choice.forceOutputIndex >= 0) {
        size_t index = static_cast<size_t>(choice.forceOutputIndex);
        CUdeviceptr forcePointer = getOpaqueDeviceMemoryDataPointer(session,
                outputBuffers[index], string(choice.label) + " force");
        result.forceOutput = OpenMmPjrtForceOutput(&outputLifetime,
                std::move(outputBuffers[index]), forcePointer);
    }

    return result;
}

// Insert a dependency so that the PJRT input stream waits
// for the OpenMM event that signals input buffers are fully written.
void PjrtRuntime::waitOnStream(CUstream stream, CUevent readyEvent) {
    if (stream != nullptr && readyEvent != nullptr) {
        CUresult result = cuStreamWaitEvent(stream, readyEvent, 0);
        if (result != CUDA_SUCCESS)
            throw runtime_error(
                    "JaxForce PJRT: failed to make PJRT input stream wait for OpenMM packed inputs");
    }
}
