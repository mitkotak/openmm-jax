#ifndef OPENMM_JAX_FORCE_PROXY_H_
#define OPENMM_JAX_FORCE_PROXY_H_

#include "openmm/serialization/SerializationProxy.h"

namespace OpenMM {

class JaxForceProxy : public SerializationProxy {
public:
    JaxForceProxy();
    void serialize(const void* object, SerializationNode& node) const override;
    void* deserialize(const SerializationNode& node) const override;
};

} // namespace OpenMM

#endif
