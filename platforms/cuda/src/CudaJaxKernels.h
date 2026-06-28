#ifndef OPENMM_CUDA_JAX_KERNELS_H_
#define OPENMM_CUDA_JAX_KERNELS_H_

#include "pjrt/CudaEvent.h"
#include "pjrt/CudaPrimaryContextGuard.h"
#include "pjrt/PjrtRuntime.h"
#include "JaxKernels.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaContext.h"
#include <cuda.h>
#include <cstddef>
#include <string>

namespace JaxPlugin {

class CudaCalcJaxForceKernel : public CalcJaxForceKernel {
public:
    CudaCalcJaxForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu);
    ~CudaCalcJaxForceKernel();
    void initialize(const OpenMM::System& system, const JaxForce& force) override;
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;

private:
    OpenMM::CudaContext& cu;
    OpenMM::CudaArray packedPositions;
    OpenMM::CudaArray boxVectors;
    int numParticles;
    bool usePeriodic;
    bool outputsForces;
    CUfunction copyInputsKernel;
    CUfunction addForcesKernel;
    PrimaryContextRetainer primaryContext;
    PjrtRuntime pjrtRuntime;

    RecordedCudaEvent prepareJaxInputs(CUstream openmmStream);
    void addForces(CUdeviceptr forcePointer);
};

class CudaCalcPythonJaxForceKernel : public CalcPythonJaxForceKernel {
public:
    CudaCalcPythonJaxForceKernel(std::string name, const OpenMM::Platform& platform,
            OpenMM::CudaContext& cu);
    ~CudaCalcPythonJaxForceKernel();
    void initialize(OpenMM::ContextImpl& context, const PythonJaxForce& force) override;
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;

private:
    OpenMM::CudaContext& cu;
    const PythonJaxForceComputation* energyComputation;
    const PythonJaxForceComputation* forcesComputation;
    const PythonJaxForceComputation* energyAndForcesComputation;
    OpenMM::CudaArray packedPositions;
    OpenMM::CudaArray boxVectors;
    int numParticles;
    bool usePeriodic;
    CUfunction copyInputsKernel;
    CUfunction addForcesKernel;

    void prepareInputs(CUstream openmmStream);
    void addForces(CUdeviceptr forcePointer);
};

} // namespace JaxPlugin

#endif
