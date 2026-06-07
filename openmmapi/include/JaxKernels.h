#ifndef OPENMM_JAX_KERNELS_H_
#define OPENMM_JAX_KERNELS_H_

#include "JaxForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace JaxPlugin {

class CalcJaxForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcJaxForce";
    }
    CalcJaxForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }

    virtual void initialize(const OpenMM::System& system, const JaxForce& force) = 0;
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

} // namespace JaxPlugin

#endif
