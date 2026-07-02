#include "CudaJaxKernels.h"
#include "CudaJaxKernelSources.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include <cuda.h>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

class CudaCalcJaxForceKernel::ReorderListener : public ComputeContext::ReorderListener {
public:
    ReorderListener(CudaCalcJaxForceKernel& owner) : owner(owner) {
    }
    void execute() override {
        owner.uploadSelectedParticles();
    }
private:
    CudaCalcJaxForceKernel& owner;
};

CudaCalcJaxForceKernel::CudaCalcJaxForceKernel(string name, const Platform& platform, CudaContext& cu) :
        CalcJaxForceKernel(name, platform), cu(cu), numSystemParticles(0), numJaxParticles(0),
        useParticleSubset(false), usePeriodic(false), primaryContext(cu) {
}

CudaCalcJaxForceKernel::~CudaCalcJaxForceKernel() {
    try {
        ScopedPrimaryContext pjrtContext(cu, primaryContext.get());
        pjrtRuntime.close();
        pjrtContext.pop();
    } catch (...) {}
}

void CudaCalcJaxForceKernel::initialize(const System& system, const JaxForce& force) {
    numSystemParticles = system.getNumParticles();
    particles = validateAndCopyParticles(system, force);
    useParticleSubset = !particles.empty();
    numJaxParticles = (useParticleSubset ? static_cast<int>(particles.size()) : numSystemParticles);
    if (numJaxParticles == 0)
        throw OpenMMException("JaxForce: at least one particle must be selected");
    usePeriodic = force.usesPeriodicBoundaryConditions();

    {
        ContextSelector selector(cu);
        map<string, string> defines;
        CUmodule program = cu.createModule(CudaJaxKernelSources::jaxForce, defines);
        copyInputsAllKernel = cu.getKernel(program, "copyInputsAll");
        copyInputsSubsetKernel = cu.getKernel(program, "copyInputsSubset");
        addForcesAllKernel = cu.getKernel(program, "addForcesAll");
        addForcesSubsetKernel = cu.getKernel(program, "addForcesSubset");
        int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
        packedPositions.initialize(cu, 3*numJaxParticles, elementSize, "jaxPackedPositions");
        boxVectors.initialize(cu, 9, elementSize, "jaxBoxVectors");
        if (useParticleSubset) {
            selectedParticles.initialize(cu, numJaxParticles, sizeof(int), "jaxSelectedParticles");
            uploadSelectedParticles();
            cu.addReorderListener(new ReorderListener(*this));
        }
    }

    ScopedPrimaryContext pjrtContext(cu, primaryContext.get());
    pjrtRuntime.initialize(force.getPjrtPluginPath(), force.getForceMlir(),
            force.getEnergyMlir(), force.getEnergyAndForcesMlir(),
            force.getCompileOptions());
    pjrtContext.pop();
}

vector<int> CudaCalcJaxForceKernel::validateAndCopyParticles(const System& system,
        const JaxForce& force) const {
    int numParticles = system.getNumParticles();
    const vector<int>& particles = force.getParticles();
    if (particles.empty())
        return {};

    vector<bool> seen(numParticles, false);
    for (size_t i = 0; i < particles.size(); i++) {
        int particle = particles[i];
        if (particle < 0 || particle >= numParticles) {
            stringstream message;
            message << "JaxForce: particle index " << particle
                    << " is outside the System particle range [0, "
                    << numParticles << ")";
            throw OpenMMException(message.str());
        }
        if (seen[particle])
            throw OpenMMException("JaxForce: particle indices must be unique");
        seen[particle] = true;
    }
    return particles;
}

void CudaCalcJaxForceKernel::uploadSelectedParticles() {
    ContextSelector selector(cu);
    const vector<int>& order = cu.getAtomIndex();
    vector<int> inverseOrder(numSystemParticles);
    if (order.size() < static_cast<size_t>(numSystemParticles)) {
        selectedParticles.upload(particles);
        return;
    }
    for (int i = 0; i < numSystemParticles; i++) {
        int systemParticle = order[i];
        if (systemParticle < 0 || systemParticle >= numSystemParticles) {
            selectedParticles.upload(particles);
            return;
        }
        inverseOrder[systemParticle] = i;
    }
    vector<int> reordered(particles.size());
    for (size_t i = 0; i < particles.size(); i++)
        reordered[i] = inverseOrder[particles[i]];
    selectedParticles.upload(reordered);
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
    void* selected = nullptr;
    void* args[9];
    CUfunction kernel;
    int numKernelParticles;
    if (useParticleSubset) {
        CUdeviceptr selectedPointer = selectedParticles.getDevicePointer();
        selected = reinterpret_cast<void*>(selectedPointer);
        args[0] = &packed;
        args[1] = &box;
        args[2] = &posq;
        args[3] = &selected;
        args[4] = &numJaxParticles;
        args[5] = &usePeriodic;
        args[6] = cu.getPeriodicBoxVecXPointer();
        args[7] = cu.getPeriodicBoxVecYPointer();
        args[8] = cu.getPeriodicBoxVecZPointer();
        kernel = copyInputsSubsetKernel;
        numKernelParticles = numJaxParticles;
    }
    else {
        args[0] = &packed;
        args[1] = &box;
        args[2] = &posq;
        args[3] = &atomIndex;
        args[4] = &numSystemParticles;
        args[5] = &usePeriodic;
        args[6] = cu.getPeriodicBoxVecXPointer();
        args[7] = cu.getPeriodicBoxVecYPointer();
        args[8] = cu.getPeriodicBoxVecZPointer();
        kernel = copyInputsAllKernel;
        numKernelParticles = numSystemParticles;
    }
    const int blockSize = 256;
    int gridSize = (numKernelParticles+blockSize-1)/blockSize;
    CHECK_RESULT(cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, 0, openmmStream, args, nullptr),
            "Failed to launch JAX input copy kernel");

    return RecordedCudaEvent::record(openmmStream,
            "Failed to create JAX input readiness event",
            "Failed to record JAX input readiness event");
}

void CudaCalcJaxForceKernel::addForces(CUdeviceptr forcePointer) {
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    CUdeviceptr forceBufferPointer = cu.getForce().getDevicePointer();
    CUdeviceptr atomIndexPointer = cu.getAtomIndexArray().getDevicePointer();
    void* forces = reinterpret_cast<void*>(forcePointer);
    void* forceBuffer = reinterpret_cast<void*>(forceBufferPointer);
    void* atomIndex = reinterpret_cast<void*>(atomIndexPointer);
    if (useParticleSubset) {
        CUdeviceptr selectedPointer = selectedParticles.getDevicePointer();
        void* selected = reinterpret_cast<void*>(selectedPointer);
        void* args[] = {&forces,
                        &forceBuffer,
                        &selected,
                        &numJaxParticles,
                        &paddedNumAtoms};
        cu.executeKernel(addForcesSubsetKernel, args, numJaxParticles);
    }
    else {
        void* args[] = {&forces,
                        &forceBuffer,
                        &atomIndex,
                        &numSystemParticles,
                        &paddedNumAtoms};
        cu.executeKernel(addForcesAllKernel, args, numSystemParticles);
    }
}

double CudaCalcJaxForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!includeForces && !includeEnergy)
        return 0.0;
    CUstream openmmStream;
    RecordedCudaEvent inputReadyEvent;
    {
        ContextSelector selector(cu);
        openmmStream = cu.getCurrentStream();
        inputReadyEvent = prepareJaxInputs(openmmStream);
    }
    OpenMmPjrtInputs inputs;
    inputs.positions = packedPositions.getDevicePointer();
    inputs.boxVectors = boxVectors.getDevicePointer();
    inputs.numInputParticles = numJaxParticles;
    inputs.deviceIndex = cu.getDeviceIndex();
    inputs.stream = openmmStream;
    inputs.usePeriodic = usePeriodic;
    inputs.inputReadyEvent = inputReadyEvent.get();
    inputs.useDoublePrecisionReal = cu.getUseDoublePrecision();
    ScopedPrimaryContext pjrtContext(cu, primaryContext.get());
    OpenMmPjrtExecutionResult result = pjrtRuntime.execute(inputs, includeForces, includeEnergy);
    pjrtContext.pop();

    if (includeForces) {
        ContextSelector selector(cu);
        result.forceOutput.consumeOnStream(openmmStream,
                [this](CUdeviceptr fp) { addForces(fp); });
    }

    return result.energy;
}
