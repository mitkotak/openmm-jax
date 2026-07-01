#ifndef OPENMM_JAX_FORCE_H_
#define OPENMM_JAX_FORCE_H_

#include "openmm/Force.h"
#include <string>
#include <vector>

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

    void setPjrtPluginPath(const std::string& path);
    const std::string& getPjrtPluginPath() const;

    /**
     * Set the particles this force applies to.
     *
     * If this is empty, all particles in the System are used.  Otherwise, the
     * exported JAX functions receive positions only for these particles, in
     * this order, and must return forces with the same shape and order.
     */
    void setParticles(const std::vector<int>& particles);
    const std::vector<int>& getParticles() const;

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    std::string forceMlir;
    std::string energyMlir;
    std::string energyAndForcesMlir;
    std::string compileOptions;
    std::string pjrtPluginPath;
    std::vector<int> particles;

    bool usePeriodic;
};

} // namespace JaxPlugin

#endif
