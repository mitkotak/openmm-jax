#ifndef OPENMM_JAX_FORCE_IMPL_H_
#define OPENMM_JAX_FORCE_IMPL_H_

#include "JaxForce.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"
#include <map>
#include <string>
#include <vector>

namespace JaxPlugin {

class JaxForceImpl : public OpenMM::ForceImpl {
public:
    explicit JaxForceImpl(const JaxForce& owner);
    void initialize(OpenMM::ContextImpl& context) override;
    const JaxForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) override {
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups) override;
    std::map<std::string, double> getDefaultParameters() override;
    std::vector<std::string> getKernelNames() override;

private:
    const JaxForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace JaxPlugin

#endif
