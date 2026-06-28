#ifndef OPENMM_PYTHON_JAX_FORCE_IMPL_H_
#define OPENMM_PYTHON_JAX_FORCE_IMPL_H_

#include "PythonJaxForce.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"
#include <map>
#include <string>
#include <vector>

namespace JaxPlugin {

class PythonJaxForceImpl : public OpenMM::ForceImpl {
public:
    explicit PythonJaxForceImpl(const PythonJaxForce& owner);
    void initialize(OpenMM::ContextImpl& context) override;
    const PythonJaxForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) override {
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces,
            bool includeEnergy, int groups) override;
    std::map<std::string, double> getDefaultParameters() override;
    std::vector<std::string> getKernelNames() override;

private:
    const PythonJaxForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace JaxPlugin

#endif
