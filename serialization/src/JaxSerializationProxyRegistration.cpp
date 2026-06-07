#include "JaxForce.h"
#include "JaxForceProxy.h"
#include "openmm/serialization/SerializationProxy.h"

extern "C" void __attribute__((constructor)) registerJaxSerializationProxies();

using namespace JaxPlugin;
using namespace OpenMM;

extern "C" void registerJaxSerializationProxies() {
    SerializationProxy::registerProxy(typeid(JaxForce), new JaxForceProxy());
}
