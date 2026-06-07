#include "JaxForceProxy.h"
#include "JaxForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/SerializationNode.h"
#include <algorithm>

using namespace JaxPlugin;
using namespace OpenMM;
using namespace std;

namespace {

string normalizedPath(const string& path) {
    string normalized = path;
    replace(normalized.begin(), normalized.end(), '\\', '/');
    return normalized;
}

bool hasSuffix(const string& value, const string& suffix) {
    return value.size() >= suffix.size() &&
            value.compare(value.size()-suffix.size(), suffix.size(), suffix) == 0;
}

string getPjrtPluginSpec(const string& pluginPath) {
    const string cuda13Spec = "jax_plugins.xla_cuda13";
    const string cuda12Spec = "jax_plugins.xla_cuda12";
    if (pluginPath == cuda13Spec || pluginPath == cuda12Spec)
        return pluginPath;

    string path = normalizedPath(pluginPath);
    if (hasSuffix(path, "jax_plugins/xla_cuda13/xla_cuda_plugin.so"))
        return cuda13Spec;
    if (hasSuffix(path, "jax_plugins/xla_cuda12/xla_cuda_plugin.so"))
        return cuda12Spec;
    return pluginPath;
}

}

JaxForceProxy::JaxForceProxy() : SerializationProxy("JaxForce") {
}

void JaxForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 15);
    const JaxForce& force = *reinterpret_cast<const JaxForce*>(object);
    node.setStringProperty("forceMlir", force.getForceMlir());
    node.setStringProperty("energyMlir", force.getEnergyMlir());
    node.setStringProperty("energyAndForcesMlir", force.getEnergyAndForcesMlir());
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
    node.setBoolProperty("outputsForces", force.getOutputsForces());
    node.setStringProperty("pjrtPluginSpec", getPjrtPluginSpec(force.getPjrtPluginPath()));
    node.setStringProperty("compileOptions", force.getCompileOptionsBase64());
}

void* JaxForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version != 15)
        throw OpenMMException("JaxForceProxy: Unsupported version number (expected 15)");
    string forceMlir = node.getStringProperty("forceMlir");
    string energyMlir = node.getStringProperty("energyMlir");
    string energyAndForcesMlir = node.getStringProperty("energyAndForcesMlir");

    string compileOptions = node.getStringProperty("compileOptions");
    JaxForce* force = new JaxForce(forceMlir, energyMlir, energyAndForcesMlir, compileOptions);
    force->setForceGroup(node.getIntProperty("forceGroup"));
    force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));
    force->setOutputsForces(node.getBoolProperty("outputsForces"));
    string pjrtPlugin = node.hasProperty("pjrtPluginSpec") ?
            node.getStringProperty("pjrtPluginSpec") :
            node.getStringProperty("pjrtPluginPath");
    force->setPjrtPluginPath(pjrtPlugin);
    return force;
}
