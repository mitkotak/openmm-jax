#include "JaxForceProxy.h"
#include "JaxForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/SerializationNode.h"

using namespace JaxPlugin;
using namespace OpenMM;
using namespace std;

namespace {

const int CurrentVersion = 16;

string getSerializedPjrtPluginPath(const SerializationNode& node) {
    if (node.hasProperty("pjrtPluginPath"))
        return node.getStringProperty("pjrtPluginPath");
    throw OpenMMException("JaxForceProxy: Missing required pjrtPluginPath property");
}

} // namespace

JaxForceProxy::JaxForceProxy() : SerializationProxy("JaxForce") {
}

void JaxForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", CurrentVersion);
    const JaxForce& force = *reinterpret_cast<const JaxForce*>(object);
    node.setStringProperty("forceMlir", force.getForceMlir());
    node.setStringProperty("energyMlir", force.getEnergyMlir());
    node.setStringProperty("energyAndForcesMlir", force.getEnergyAndForcesMlir());
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
    node.setBoolProperty("outputsForces", force.getOutputsForces());
    node.setStringProperty("pjrtPluginPath", force.getPjrtPluginPath());
    node.setStringProperty("compileOptions", force.getCompileOptionsBase64());
}

void* JaxForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version != CurrentVersion)
        throw OpenMMException("JaxForceProxy: Unsupported version number");
    string forceMlir = node.getStringProperty("forceMlir");
    string energyMlir = node.getStringProperty("energyMlir");
    string energyAndForcesMlir = node.getStringProperty("energyAndForcesMlir");

    string compileOptions = node.getStringProperty("compileOptions");
    string pjrtPlugin = getSerializedPjrtPluginPath(node);
    JaxForce* force = new JaxForce(forceMlir, energyMlir, energyAndForcesMlir, compileOptions);
    force->setForceGroup(node.getIntProperty("forceGroup"));
    force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));
    force->setOutputsForces(node.getBoolProperty("outputsForces"));
    force->setPjrtPluginPath(pjrtPlugin);
    return force;
}
