#include "JaxForceProxy.h"
#include "JaxForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/SerializationNode.h"

using namespace JaxPlugin;
using namespace OpenMM;
using namespace std;

namespace {

string getSerializedPjrtPluginPath(const SerializationNode& node) {
    if (node.hasProperty("pjrtPluginPath"))
        return node.getStringProperty("pjrtPluginPath");
    throw OpenMMException("JaxForceProxy: Missing required pjrtPluginPath property");
}

} // namespace

JaxForceProxy::JaxForceProxy() : SerializationProxy("JaxForce") {
}

void JaxForceProxy::serialize(const void* object, SerializationNode& node) const {
    const JaxForce& force = *reinterpret_cast<const JaxForce*>(object);
    node.setStringProperty("forceMlir", force.getForceMlir());
    node.setStringProperty("energyMlir", force.getEnergyMlir());
    node.setStringProperty("energyAndForcesMlir", force.getEnergyAndForcesMlir());
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
    node.setStringProperty("pjrtPluginPath", force.getPjrtPluginPath());
    node.setStringProperty("compileOptions", force.getCompileOptionsBase64());
    SerializationNode& particles = node.createChildNode("particles");
    for (int particle : force.getParticles())
        particles.createChildNode("particle").setIntProperty("index", particle);
}

void* JaxForceProxy::deserialize(const SerializationNode& node) const {
    string forceMlir = node.getStringProperty("forceMlir");
    string energyMlir = node.getStringProperty("energyMlir");
    string energyAndForcesMlir = node.getStringProperty("energyAndForcesMlir");

    string compileOptions = node.getStringProperty("compileOptions");
    string pjrtPlugin = getSerializedPjrtPluginPath(node);
    JaxForce* force = new JaxForce(forceMlir, energyMlir, energyAndForcesMlir, compileOptions);
    force->setForceGroup(node.getIntProperty("forceGroup"));
    force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));
    force->setPjrtPluginPath(pjrtPlugin);
    vector<int> particles;
    for (const SerializationNode& particle : node.getChildNode("particles").getChildren())
        particles.push_back(particle.getIntProperty("index"));
    force->setParticles(particles);
    return force;
}
