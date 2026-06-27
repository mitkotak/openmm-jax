#ifndef OPENMM_PJRT_CLIENT_SESSION_H_
#define OPENMM_PJRT_CLIENT_SESSION_H_

#include "PjrtHandles.h"
#include "PjrtPlugin.h"
#include <cuda.h>
#include <string>
#include <vector>

struct PJRT_Stream_Extension;

namespace JaxPlugin {

class PjrtClientSession {
public:
    PjrtClientSession() = default;
    PjrtClientSession(const PjrtClientSession&) = delete;
    PjrtClientSession& operator=(const PjrtClientSession&) = delete;
    PjrtClientSession& operator=(PjrtClientSession&& other);
    ~PjrtClientSession();

    void initialize(const std::string& pluginPath);
    void close();

    const PJRT_Api* api() const;
    PjrtPluginLibrary& pluginLibrary();
    PJRT_Client* client() const;
    PJRT_Device* device(int deviceIndex) const;
    bool hasDevice(int deviceIndex) const;
    CUstream getStreamForExternalReadyEvents(int deviceIndex);

private:
    void clearClientState();

    PjrtPluginLibrary library;
    PjrtClientPtr clientHandle;
    std::vector<PJRT_Device*> devices;
    PJRT_Stream_Extension* streamExtension = nullptr;
};

} // namespace JaxPlugin

#endif
