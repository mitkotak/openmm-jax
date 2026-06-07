#ifndef OPENMM_CUDA_JAX_KERNEL_FACTORY_H_
#define OPENMM_CUDA_JAX_KERNEL_FACTORY_H_

#include "openmm/KernelFactory.h"
#include "openmm/cuda/CudaPlatform.h"

namespace JaxPlugin {

class CudaJaxKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform,
                                         OpenMM::ContextImpl& context) const override;
};

} // namespace JaxPlugin

#endif
