#include "PythonJaxForce.h"
#include "internal/PythonJaxForceImpl.h"

using namespace JaxPlugin;
using namespace OpenMM;
using namespace std;

PythonJaxForce::PythonJaxForce(PythonJaxForceComputation* energyComputation,
        PythonJaxForceComputation* forcesComputation,
        PythonJaxForceComputation* energyAndForcesComputation,
        const map<string, double>& globalParameters) :
        energyComputation(energyComputation), forcesComputation(forcesComputation),
        energyAndForcesComputation(energyAndForcesComputation),
        globalParameters(globalParameters), usePeriodic(false) {
}

PythonJaxForce::~PythonJaxForce() {
    delete energyComputation;
    delete forcesComputation;
    delete energyAndForcesComputation;
}

const PythonJaxForceComputation& PythonJaxForce::getEnergyComputation() const {
    return *energyComputation;
}

const PythonJaxForceComputation& PythonJaxForce::getForcesComputation() const {
    return *forcesComputation;
}

const PythonJaxForceComputation& PythonJaxForce::getEnergyAndForcesComputation() const {
    return *energyAndForcesComputation;
}

const map<string, double>& PythonJaxForce::getGlobalParameters() const {
    return globalParameters;
}

const vector<char>& PythonJaxForce::getPickledEnergyFunction() const {
    return pickledEnergy;
}

const vector<char>& PythonJaxForce::getPickledForcesFunction() const {
    return pickledForces;
}

const vector<char>& PythonJaxForce::getPickledEnergyAndForcesFunction() const {
    return pickledEnergyAndForces;
}

void PythonJaxForce::setPickledEnergyFunction(char* function, int length) {
    pickledEnergy = vector<char>(function, function+length);
}

void PythonJaxForce::setPickledForcesFunction(char* function, int length) {
    pickledForces = vector<char>(function, function+length);
}

void PythonJaxForce::setPickledEnergyAndForcesFunction(char* function, int length) {
    pickledEnergyAndForces = vector<char>(function, function+length);
}

void PythonJaxForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

bool PythonJaxForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}

ForceImpl* PythonJaxForce::createImpl() const {
    return new PythonJaxForceImpl(*this);
}
