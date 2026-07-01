#include "PjrtRuntime.h"
#include "PjrtBufferInterop.h"
#include "PjrtLoadedExecutable.h"
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace JaxPlugin;
using namespace std;

namespace {

bool parseStablehloFirstInputUsesDoublePrecisionReal(const string& mlir, const string& label) {
    size_t mainFunction = mlir.find("@main");
    if (mainFunction == string::npos)
        throw runtime_error("JaxForce PJRT: " + label + " StableHLO program has no @main function");
    size_t argumentList = mlir.find('(', mainFunction);
    if (argumentList == string::npos)
        throw runtime_error("JaxForce PJRT: " + label + " StableHLO @main function has no arguments");
    size_t tensorStart = mlir.find("tensor<", argumentList);
    if (tensorStart == string::npos)
        throw runtime_error(
                "JaxForce PJRT: could not determine " + label +
                " StableHLO input precision; expected a tensor argument");
    size_t tensorEnd = mlir.find('>', tensorStart);
    if (tensorEnd == string::npos)
        throw runtime_error(
                "JaxForce PJRT: could not determine " + label +
                " StableHLO input precision; malformed tensor argument");

    string tensorType = mlir.substr(tensorStart, tensorEnd - tensorStart + 1);
    if (tensorType.find("xf64") != string::npos)
        return true;
    if (tensorType.find("xf32") != string::npos)
        return false;
    throw runtime_error(
            "JaxForce PJRT: unsupported " + label +
            " StableHLO input precision in " + tensorType +
            " (expected f32 or f64 positions)");
}

const char* precisionName(bool useDoublePrecisionReal) {
    return useDoublePrecisionReal ? "double/f64" : "single-or-mixed/f32";
}

void checkStablehloPrecisionMatches(const string& label, bool expected, bool actual) {
    if (expected == actual)
        return;
    throw runtime_error(
            "JaxForce PJRT: StableHLO input precision mismatch between programs: " +
            label + " uses " + precisionName(actual) + " positions, but earlier "
            "programs use " + precisionName(expected) + " positions");
}

RequestedOutputs getRequestedOutputs(bool includeForces, bool includeEnergy) {
    if (includeForces && includeEnergy)
        return RequestedOutputs::EnergyAndForces;
    if (includeForces)
        return RequestedOutputs::Forces;
    return RequestedOutputs::Energy;
}

void makePjrtStreamWaitForOpenMmInputs(CUstream stream, CUevent readyEvent) {
    if (stream == nullptr || readyEvent == nullptr)
        return;
    CUresult result = cuStreamWaitEvent(stream, readyEvent, 0);
    if (result != CUDA_SUCCESS)
        throw runtime_error(
                "JaxForce PJRT: failed to make PJRT input stream wait for OpenMM packed inputs");
}

} // namespace

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
    stablehloUsesDoublePrecisionReal = false;
}

void PjrtRuntime::initialize(const string& pluginPath, const string& forceMlir,
        const string& energyMlir, const string& energyAndForcesMlir,
        const string& compileOptions) {
    close();

    if (forceMlir.empty() || energyMlir.empty() || energyAndForcesMlir.empty())
        throw runtime_error(
                "JaxForce PJRT: force, energy, and energy+forces StableHLO programs must all be provided");
    bool forceUsesDoublePrecisionReal =
            parseStablehloFirstInputUsesDoublePrecisionReal(forceMlir, "force");
    bool energyUsesDoublePrecisionReal =
            parseStablehloFirstInputUsesDoublePrecisionReal(energyMlir, "energy");
    bool energyAndForcesUsesDoublePrecisionReal =
            parseStablehloFirstInputUsesDoublePrecisionReal(energyAndForcesMlir, "energy+forces");
    checkStablehloPrecisionMatches("energy", forceUsesDoublePrecisionReal,
            energyUsesDoublePrecisionReal);
    checkStablehloPrecisionMatches("energy+forces", forceUsesDoublePrecisionReal,
            energyAndForcesUsesDoublePrecisionReal);

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
    stablehloUsesDoublePrecisionReal = forceUsesDoublePrecisionReal;
}


OpenMmPjrtExecutionResult PjrtRuntime::execute(
        const OpenMmPjrtInputs& pjrtInputs, bool includeForces,
        bool includeEnergy) {
    if (!includeForces && !includeEnergy)
        return {};

    outputLifetime.cleanupBeforeExecution();

    RequestedOutputs requested = getRequestedOutputs(includeForces, includeEnergy);
    SelectedPjrtProgram program = selectProgram(requested);

    validatePrecision(pjrtInputs);

    PjrtInputBuffers inputBuffers = createInputViews(pjrtInputs, program);
    PjrtOutputBuffers outputBuffers = executeLoadedExecutable(session,
            program, inputBuffers, pjrtInputs.deviceIndex);
    awaitDeviceCompleteEvent(session, outputBuffers.completeEvent, program.label);

    return consumeOutputs(std::move(outputBuffers), program, pjrtInputs);
}

SelectedPjrtProgram PjrtRuntime::selectProgram(RequestedOutputs outputs) const {
    switch (outputs) {
        case RequestedOutputs::EnergyAndForces:
            return {energyAndForcesExecutable.get(), 2, 1, 0, "energy+forces"};
        case RequestedOutputs::Forces:
            return {forceExecutable.get(), 1, 0, -1, "force"};
        case RequestedOutputs::Energy:
            return {energyExecutable.get(), 1, -1, 0, "energy"};
    }
    throw runtime_error("JaxForce PJRT: unknown requested output set");
}

void PjrtRuntime::validatePrecision(const OpenMmPjrtInputs& inputs) const {
    if (inputs.useDoublePrecisionReal != stablehloUsesDoublePrecisionReal) {
        stringstream message;
        message << "JaxForce PJRT: OpenMM CUDA context precision does not match "
                << "the JaxForce StableHLO input precision. OpenMM is providing "
                << precisionName(inputs.useDoublePrecisionReal)
                << " position buffers, but the JaxForce was exported for "
                << precisionName(stablehloUsesDoublePrecisionReal)
                << " position buffers. OpenMM-JAX does not implicitly convert "
                << "between f32 and f64 buffers; configure both sides consistently "
                << "(for example, pass precision=\"double\" or use_float64=True "
                << "when using CUDA Precision=\"double\", or use CUDA "
                << "Precision=\"mixed\"/\"single\" with an f32 JaxForce).";
        throw runtime_error(message.str());
    }
}

PjrtInputBuffers PjrtRuntime::createInputViews(const OpenMmPjrtInputs& inputs,
        const SelectedPjrtProgram& program) {
    CUstream inputStream = session.getStreamForExternalReadyEvents(inputs.deviceIndex);
    makePjrtStreamWaitForOpenMmInputs(inputStream, inputs.inputReadyEvent);

    PjrtInputBuffers inputBuffers;
    PJRT_Buffer_Type inputType = inputs.useDoublePrecisionReal ?
            PJRT_Buffer_Type_F64 : PJRT_Buffer_Type_F32;
    int64_t positionDims[2] = {inputs.numInputParticles, 3};
    inputBuffers.push(createViewOfDeviceBuffer(session,
            inputs.positions, positionDims, 2,
            inputType, inputStream, inputs.deviceIndex,
            string(program.label) + " positions"));
    if (inputs.usePeriodic) {
        int64_t boxDims[2] = {3, 3};
        inputBuffers.push(createViewOfDeviceBuffer(session,
                inputs.boxVectors, boxDims, 2,
                inputType, inputStream, inputs.deviceIndex,
                string(program.label) + " boxVectors"));
    }
    return inputBuffers;
}

OpenMmPjrtExecutionResult PjrtRuntime::consumeOutputs(PjrtOutputBuffers outputs,
        const SelectedPjrtProgram& program,
        const OpenMmPjrtInputs& inputs) {
    OpenMmPjrtExecutionResult result;

    if (program.energyOutputIndex >= 0) {
        size_t index = static_cast<size_t>(program.energyOutputIndex);
        if (index >= program.outputCount)
            throw runtime_error("JaxForce PJRT: energy output index out of range for " +
                    string(program.label));
        CUdeviceptr energyPointer = getOpaqueDeviceMemoryDataPointer(session,
                outputs.buffers[index], string(program.label) + " energy");
        double energyDouble = 0.0;
        float energyFloat = 0.0f;
        void* energyHost = inputs.useDoublePrecisionReal ?
                static_cast<void*>(&energyDouble) : static_cast<void*>(&energyFloat);
        size_t energySize = inputs.useDoublePrecisionReal ? sizeof(double) : sizeof(float);
        CUresult copyResult = cuMemcpyDtoHAsync(
                energyHost, energyPointer, energySize, inputs.stream);
        if (copyResult != CUDA_SUCCESS)
            throw runtime_error("JaxForce PJRT: failed to copy energy scalar from device");
        CUresult syncResult = cuStreamSynchronize(inputs.stream);
        if (syncResult != CUDA_SUCCESS)
            throw runtime_error("JaxForce PJRT: failed to synchronize energy scalar copy");
        result.energy = inputs.useDoublePrecisionReal ?
                energyDouble : static_cast<double>(energyFloat);
        outputs.buffers[index].reset();
    }

    if (program.forceOutputIndex >= 0) {
        size_t index = static_cast<size_t>(program.forceOutputIndex);
        if (index >= program.outputCount)
            throw runtime_error("JaxForce PJRT: force output index out of range for " +
                    string(program.label));
        CUdeviceptr forcePointer = getOpaqueDeviceMemoryDataPointer(session,
                outputs.buffers[index], string(program.label) + " force");
        result.forceOutput = OpenMmPjrtForceOutput(&outputLifetime,
                std::move(outputs.buffers[index]), forcePointer);
    }

    return result;
}
