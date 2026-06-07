#ifndef OPENMM_PJRT_PLUGIN_H_
#define OPENMM_PJRT_PLUGIN_H_

#include "pjrt_c_api.h"
#include <string>

namespace JaxPlugin {


class PjrtPluginLibrary {
public:
    PjrtPluginLibrary();
    PjrtPluginLibrary(const PjrtPluginLibrary&) = delete;
    PjrtPluginLibrary& operator=(const PjrtPluginLibrary&) = delete;
    PjrtPluginLibrary(PjrtPluginLibrary&& other) noexcept;
    PjrtPluginLibrary& operator=(PjrtPluginLibrary&& other) noexcept;
    ~PjrtPluginLibrary();

    /** Load a PJRT plugin from the given filesystem path via dlopen. */
    void open(const std::string& path);

    /** Unload the plugin and clear all internal state. */
    void close();

    /** Return the PJRT C API function-pointer table, or null if not loaded. */
    const PJRT_Api* getApi() const;

    /**
     * Walk the extension linked-list in the PJRT_Api and return the first
     * extension matching the requested type, or null if none is found.
     */
    template <typename ExtensionType>
    ExtensionType* findExtension(PJRT_Extension_Type type) const {
        return reinterpret_cast<ExtensionType*>(findRawExtension(type));
    }

    void checkError(PJRT_Error* error, const std::string& operation) const;

private:
    PJRT_Extension_Base* findRawExtension(PJRT_Extension_Type type) const;
    void* library;
    const PJRT_Api* api;
    std::string pluginPath;
};

} // namespace JaxPlugin

#endif
