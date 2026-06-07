#ifndef OPENMM_PJRT_HANDLES_H_
#define OPENMM_PJRT_HANDLES_H_

#include "pjrt_c_api.h"
#include <memory>
#include <string>

namespace JaxPlugin {


std::string getErrorMessageAndDestroy(const PJRT_Api* api, PJRT_Error* error);


struct PjrtBufferDeleter {
    const PJRT_Api* api = nullptr;
    void operator()(PJRT_Buffer* buffer) const;
};

struct PjrtEventDeleter {
    const PJRT_Api* api = nullptr;
    void operator()(PJRT_Event* event) const;
};

struct PjrtLoadedExecutableDeleter {
    const PJRT_Api* api = nullptr;
    void operator()(PJRT_LoadedExecutable* executable) const;
};

struct PjrtClientDeleter {
    const PJRT_Api* api = nullptr;
    void operator()(PJRT_Client* client) const;
};

struct PjrtErrorDeleter {
    const PJRT_Api* api = nullptr;
    void operator()(PJRT_Error* error) const;
};

using PjrtBufferPtr = std::unique_ptr<PJRT_Buffer, PjrtBufferDeleter>;
using PjrtEventPtr = std::unique_ptr<PJRT_Event, PjrtEventDeleter>;
using PjrtLoadedExecutablePtr = std::unique_ptr<PJRT_LoadedExecutable, PjrtLoadedExecutableDeleter>;
using PjrtClientPtr = std::unique_ptr<PJRT_Client, PjrtClientDeleter>;
using PjrtErrorPtr = std::unique_ptr<PJRT_Error, PjrtErrorDeleter>;

/** Create a PjrtBufferDeleter without external reference counting. */
PjrtBufferDeleter makeBufferDeleter(const PJRT_Api* api);
PjrtEventDeleter makeEventDeleter(const PJRT_Api* api);
PjrtLoadedExecutableDeleter makeLoadedExecutableDeleter(const PJRT_Api* api);
PjrtClientDeleter makeClientDeleter(const PJRT_Api* api);
PjrtErrorDeleter makeErrorDeleter(const PJRT_Api* api);

} // namespace JaxPlugin

#endif
