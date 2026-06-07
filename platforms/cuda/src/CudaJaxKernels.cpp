#include "CudaJaxKernels.h"
#include "CudaJaxKernelSources.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <sstream>
#include <string>
#include <utility>

using namespace JaxPlugin;
using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result, prefix)                                             \
    if (result != CUDA_SUCCESS) {                                                \
        stringstream m;                                                          \
        m << prefix << ": " << cu.getErrorString(result) << " (" << result << ")"\
          << " at " << __FILE__ << ":" << __LINE__;                             \
        throw OpenMMException(m.str());                                          \
    }

CudaCalcJaxForceKernel::CudaCalcJaxForceKernel(string name, const Platform& platform, CudaContext& cu) :
        CalcJaxForceKernel(name, platform), cu(cu), primaryContext(cu) {
}

CudaCalcJaxForceKernel::~CudaCalcJaxForceKernel() {
    try {
        ScopedPrimaryContext pjrtContext(cu, primaryContext.get());
        pjrtRuntime.close();
        pjrtContext.pop();
    } catch (...) {}
}

void CudaCalcJaxForceKernel::initialize(const System& system, const JaxForce& force) {
    numParticles = system.getNumParticles();
    usePeriodic = force.usesPeriodicBoundaryConditions();
    outputsForces = force.getOutputsForces();

    ContextSelector selector(cu);
    map<string, string> defines;
    CUmodule program = cu.createModule(CudaJaxKernelSources::jaxForce, defines);
    copyInputsKernel = cu.getKernel(program, "copyInputs");
    addForcesKernel = cu.getKernel(program, "addForces");
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    packedPositions.initialize(cu, 3*numParticles, elementSize, "jaxPackedPositions");
    boxVectors.initialize(cu, 9, elementSize, "jaxBoxVectors");

    ScopedPrimaryContext pjrtContext(cu, primaryContext.get());
    pjrtRuntime.initialize(force.getPjrtPluginPath(), force.getForceMlir(),
            force.getEnergyMlir(), force.getEnergyAndForcesMlir(),
            force.getCompileOptions());
    pjrtContext.pop();
}

RecordedCudaEvent CudaCalcJaxForceKernel::prepareJaxInputs(CUstream openmmStream) {
    CUdeviceptr packedPointer = packedPositions.getDevicePointer();
    CUdeviceptr boxVectorsPointer = boxVectors.getDevicePointer();
    CUdeviceptr posqPointer = cu.getPosq().getDevicePointer();
    CUdeviceptr atomIndexPointer = cu.getAtomIndexArray().getDevicePointer();
    void* packed = reinterpret_cast<void*>(packedPointer);
    void* box = reinterpret_cast<void*>(boxVectorsPointer);
    void* posq = reinterpret_cast<void*>(posqPointer);
    void* atomIndex = reinterpret_cast<void*>(atomIndexPointer);
    void* args[] = {&packed,
                    &box,
                    &posq,
                    &atomIndex,
                    &numParticles,
                    &usePeriodic,
                    cu.getPeriodicBoxVecXPointer(),
                    cu.getPeriodicBoxVecYPointer(),
                    cu.getPeriodicBoxVecZPointer()};
    const int blockSize = 256;
    int gridSize = (numParticles+blockSize-1)/blockSize;
    CHECK_RESULT(cuLaunchKernel(copyInputsKernel, gridSize, 1, 1, blockSize, 1, 1, 0, openmmStream, args, nullptr),
            "Failed to launch JAX input copy kernel");

    return RecordedCudaEvent::record(openmmStream,
            "Failed to create JAX input readiness event",
            "Failed to record JAX input readiness event");
}

void CudaCalcJaxForceKernel::addForces(CUdeviceptr forcePointer) {
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    // OpenMM accumulates forces.  Gradient outputs are dE/dx, which is
    // minus the forces, so change the sign during accumulation.
    int forceSign = (outputsForces ? 1 : -1);
    CUdeviceptr forceBufferPointer = cu.getForce().getDevicePointer();
    CUdeviceptr atomIndexPointer = cu.getAtomIndexArray().getDevicePointer();
    void* forces = reinterpret_cast<void*>(forcePointer);
    void* forceBuffer = reinterpret_cast<void*>(forceBufferPointer);
    void* atomIndex = reinterpret_cast<void*>(atomIndexPointer);
    void* args[] = {&forces,
                    &forceBuffer,
                    &atomIndex,
                    &numParticles,
                    &paddedNumAtoms,
                    &forceSign};
    cu.executeKernel(addForcesKernel, args, numParticles);
}

double CudaCalcJaxForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    ContextSelector selector(cu);
    if (!includeForces && !includeEnergy)
        return 0.0;
    CUstream openmmStream = cu.getCurrentStream();
    RecordedCudaEvent inputReadyEvent = prepareJaxInputs(openmmStream);
    OpenMmPjrtInputs inputs;
    inputs.positions = packedPositions.getDevicePointer();
    inputs.boxVectors = boxVectors.getDevicePointer();
    inputs.numParticles = numParticles;
    inputs.deviceIndex = cu.getDeviceIndex();
    inputs.stream = openmmStream;
    inputs.usePeriodic = usePeriodic;
    inputs.inputReadyEvent = inputReadyEvent.get();
    ScopedPrimaryContext pjrtContext(cu, primaryContext.get());
    OpenMmPjrtExecutionResult result = pjrtRuntime.execute(inputs, includeForces, includeEnergy);
    pjrtContext.pop();

    if (includeForces) {
        result.forceOutput.consumeOnStream(openmmStream,
                [this](CUdeviceptr fp) { addForces(fp); });
    }

    return result.energy;
}
