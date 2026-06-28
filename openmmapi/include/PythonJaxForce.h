#ifndef OPENMM_PYTHON_JAX_FORCE_H_
#define OPENMM_PYTHON_JAX_FORCE_H_

#include "openmm/Force.h"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace JaxPlugin {

struct PythonJaxForceComputationInputs {
    std::uintptr_t positions = 0;
    std::uintptr_t boxVectors = 0;
    int numParticles = 0;
    int deviceIndex = 0;
    bool usePeriodic = false;
    bool useDoublePrecision = false;
    std::map<std::string, double> parameters;
};

struct PythonJaxForceComputationResult {
    double energy = 0.0;
    std::uintptr_t forces = 0;
    std::shared_ptr<void> lifetime;
};

class PythonJaxForceComputation {
public:
    PythonJaxForceComputation() {
    }
    virtual ~PythonJaxForceComputation() {
    }
    virtual PythonJaxForceComputationResult compute(
            const PythonJaxForceComputationInputs& inputs,
            bool includeForces, bool includeEnergy) const = 0;
};

class PythonJaxForce : public OpenMM::Force {
public:
    explicit PythonJaxForce(PythonJaxForceComputation* energyComputation,
            PythonJaxForceComputation* forcesComputation,
            PythonJaxForceComputation* energyAndForcesComputation,
            const std::map<std::string, double>& globalParameters);
    ~PythonJaxForce();

    const PythonJaxForceComputation& getEnergyComputation() const;
    const PythonJaxForceComputation& getForcesComputation() const;
    const PythonJaxForceComputation& getEnergyAndForcesComputation() const;
    const std::map<std::string, double>& getGlobalParameters() const;

    const std::vector<char>& getPickledEnergyFunction() const;
    const std::vector<char>& getPickledForcesFunction() const;
    const std::vector<char>& getPickledEnergyAndForcesFunction() const;
    void setPickledEnergyFunction(char* function, int length);
    void setPickledForcesFunction(char* function, int length);
    void setPickledEnergyAndForcesFunction(char* function, int length);

    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    PythonJaxForceComputation* energyComputation;
    PythonJaxForceComputation* forcesComputation;
    PythonJaxForceComputation* energyAndForcesComputation;
    std::map<std::string, double> globalParameters;
    bool usePeriodic;
    std::vector<char> pickledEnergy;
    std::vector<char> pickledForces;
    std::vector<char> pickledEnergyAndForces;
};

} // namespace JaxPlugin

#endif
