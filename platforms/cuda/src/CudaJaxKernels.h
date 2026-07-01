#ifndef OPENMM_CUDA_JAX_KERNELS_H_
#define OPENMM_CUDA_JAX_KERNELS_H_

#include "pjrt/CudaEvent.h"
#include "pjrt/CudaPrimaryContextGuard.h"
#include "pjrt/PjrtRuntime.h"
#include "JaxKernels.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaContext.h"
#include <cuda.h>
#include <cstddef>
#include <string>
#include <vector>

namespace JaxPlugin {

class CudaCalcJaxForceKernel : public CalcJaxForceKernel {
public:
    CudaCalcJaxForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu);
    ~CudaCalcJaxForceKernel();
    void initialize(const OpenMM::System& system, const JaxForce& force) override;
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;

private:
    class ReorderListener;

    OpenMM::CudaContext& cu;
    OpenMM::CudaArray packedPositions;
    OpenMM::CudaArray boxVectors;
    OpenMM::CudaArray selectedParticles;
    std::vector<int> particles;
    int numSystemParticles;
    int numJaxParticles;
    bool useParticleSubset;
    bool usePeriodic;
    CUfunction copyInputsAllKernel;
    CUfunction copyInputsSubsetKernel;
    CUfunction addForcesAllKernel;
    CUfunction addForcesSubsetKernel;
    PrimaryContextRetainer primaryContext;
    PjrtRuntime pjrtRuntime;

    std::vector<int> validateAndCopyParticles(const OpenMM::System& system, const JaxForce& force) const;
    void uploadSelectedParticles();
    RecordedCudaEvent prepareJaxInputs(CUstream openmmStream);
    void addForces(CUdeviceptr forcePointer);
};

} // namespace JaxPlugin

#endif
