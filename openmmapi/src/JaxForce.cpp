// Structure borrowed from https://github.com/openmm/openmm-torch/blob/master/openmmapi/src/TorchForce.cpp

#include "JaxForce.h"
#include "internal/JaxForceImpl.h"
#include "internal/JaxBase64.h"
#include "openmm/OpenMMException.h"
#include <string>

using namespace JaxPlugin;
using namespace OpenMM;
using namespace std;

namespace {

string decodeCompileOptionsBase64(const string& compileOptionsBase64) {
    if (compileOptionsBase64.empty())
        throw OpenMMException("JaxForce: compileOptionsBase64 must be provided");
    string compileOptions = decodeBase64(compileOptionsBase64, "JaxForce compileOptions");
    if (compileOptions.empty())
        throw OpenMMException("JaxForce: decoded compile options must not be empty");
    return compileOptions;
}

}

JaxForce::JaxForce(const string& forceMlir, const string& energyMlir,
                   const string& energyAndForcesMlir,
                   const string& compileOptionsBase64) :
        forceMlir(forceMlir), energyMlir(energyMlir),
        energyAndForcesMlir(energyAndForcesMlir),
        compileOptions(decodeCompileOptionsBase64(compileOptionsBase64)),
        usePeriodic(false), outputsForces(false) {
}

const string& JaxForce::getForceMlir() const {
    return forceMlir;
}

const string& JaxForce::getEnergyMlir() const {
    return energyMlir;
}

const string& JaxForce::getEnergyAndForcesMlir() const {
    return energyAndForcesMlir;
}

const string& JaxForce::getCompileOptions() const {
    return compileOptions;
}

string JaxForce::getCompileOptionsBase64() const {
    return encodeBase64(compileOptions);
}


ForceImpl* JaxForce::createImpl() const {
    return new JaxForceImpl(*this);
}

void JaxForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

bool JaxForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}

void JaxForce::setOutputsForces(bool outputsForces) {
    this->outputsForces = outputsForces;
}

bool JaxForce::getOutputsForces() const {
    return outputsForces;
}

void JaxForce::setPjrtPluginPath(const string& path) {
    pjrtPluginPath = path;
}

const string& JaxForce::getPjrtPluginPath() const {
    return pjrtPluginPath;
}
