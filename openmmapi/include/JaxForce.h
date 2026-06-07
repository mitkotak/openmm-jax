#ifndef OPENMM_JAX_FORCE_H_
#define OPENMM_JAX_FORCE_H_

#include "openmm/Force.h"
#include <string>

namespace JaxPlugin {

class JaxForce : public OpenMM::Force {
public:
    JaxForce(const std::string& forceMlir, const std::string& energyMlir,
             const std::string& energyAndForcesMlir,
             const std::string& compileOptionsBase64);

    const std::string& getForceMlir() const;
    const std::string& getEnergyMlir() const;
    const std::string& getEnergyAndForcesMlir() const;
    const std::string& getCompileOptions() const;
    std::string getCompileOptionsBase64() const;

    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;

    /**
     * Set the output sign convention for the exported array.
     *
     * If this is true, the exported array is interpreted as forces and is
     * accumulated into OpenMM's force buffer as-is. If this is false, the
     * exported array is interpreted as gradients dE/dx and is negated during
     * accumulation.
     */
    void setOutputsForces(bool outputsForces);
    bool getOutputsForces() const;

    void setPjrtPluginPath(const std::string& path);
    const std::string& getPjrtPluginPath() const;

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    std::string forceMlir;
    std::string energyMlir;
    std::string energyAndForcesMlir;
    std::string compileOptions;
    std::string pjrtPluginPath;

    bool usePeriodic;
    bool outputsForces;
};

} // namespace JaxPlugin

#endif
