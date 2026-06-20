#include "PjrtPlugin.h"
#include "PjrtHandles.h"
#include <cstddef>
#include <dlfcn.h>
#include <stdexcept>
#include <sstream>
#include <utility>

using namespace JaxPlugin;
using namespace std;

namespace {

using GetPjrtApiFn = const PJRT_Api*();

void requireApiField(const PJRT_Api* api, size_t fieldEnd, const char* fieldName) {
    if (api == nullptr || api->struct_size < fieldEnd) {
        stringstream message;
        message << "JaxForce PJRT: plugin API table is too old; missing "
                << fieldName << " (plugin struct_size="
                << (api == nullptr ? 0 : api->struct_size)
                << ", required at least " << fieldEnd << ")";
        throw runtime_error(message.str());
    }
}

void validateApiHeader(const PJRT_Api* api) {
    if (api == nullptr)
        throw runtime_error("JaxForce PJRT: GetPjrtApi returned null");
    requireApiField(api, PJRT_STRUCT_SIZE(PJRT_Api, pjrt_api_version),
            "pjrt_api_version");
    if (api->pjrt_api_version.major_version != PJRT_API_MAJOR) {
        stringstream message;
        message << "JaxForce PJRT: incompatible PJRT C API major version "
                << api->pjrt_api_version.major_version << " from plugin; "
                << "this build expects " << PJRT_API_MAJOR;
        throw runtime_error(message.str());
    }
}

} // namespace

PjrtPluginLibrary::PjrtPluginLibrary() : library(nullptr), api(nullptr) {
}

PjrtPluginLibrary::PjrtPluginLibrary(PjrtPluginLibrary&& other) noexcept :
        library(other.library), api(other.api), pluginPath(std::move(other.pluginPath)) {
    other.library = nullptr;
    other.api = nullptr;
}

PjrtPluginLibrary& PjrtPluginLibrary::operator=(PjrtPluginLibrary&& other) noexcept {
    if (this != &other) {
        close();
        library = other.library;
        api = other.api;
        pluginPath = std::move(other.pluginPath);
        other.library = nullptr;
        other.api = nullptr;
    }
    return *this;
}

PjrtPluginLibrary::~PjrtPluginLibrary() {
    close();
}

void PjrtPluginLibrary::open(const string& path) {
    close();
    try {
        pluginPath = path;
        library = dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr)
            throw runtime_error(string("JaxForce PJRT: failed to load PJRT plugin '") + pluginPath + "': " + dlerror());
        auto getPjrtApi = reinterpret_cast<GetPjrtApiFn*>(dlsym(library, "GetPjrtApi"));
        if (getPjrtApi == nullptr)
            throw runtime_error("JaxForce PJRT: plugin is missing GetPjrtApi");
        api = getPjrtApi();
        validateApiHeader(api);
        requireApiField(api, PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Client_Devices),
                "PJRT_Client_Devices");
        if (api->PJRT_Client_Create == nullptr || api->PJRT_Client_Devices == nullptr)
            throw runtime_error("JaxForce PJRT: plugin API is missing required client entry points");
        if (api->PJRT_Plugin_Initialize != nullptr) {
            PJRT_Plugin_Initialize_Args initArgs;
            initArgs.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
            initArgs.extension_start = nullptr;
            checkError(api->PJRT_Plugin_Initialize(&initArgs), "PJRT_Plugin_Initialize");
        }
    }
    catch (...) {
        close();
        throw;
    }
}

void PjrtPluginLibrary::close() {
    api = nullptr;
    if (library != nullptr)
        dlclose(library);
    library = nullptr;
    pluginPath.clear();
}

const PJRT_Api* PjrtPluginLibrary::getApi() const {
    return api;
}

PJRT_Extension_Base* PjrtPluginLibrary::findRawExtension(PJRT_Extension_Type type) const {
    if (api == nullptr)
        return nullptr;
    PJRT_Extension_Base* starts[] = {api->extension_start, api->pjrt_api_version.extension_start};
    for (PJRT_Extension_Base* start : starts)
        for (PJRT_Extension_Base* extension = start; extension != nullptr; extension = extension->next)
            if (extension->type == type)
                return extension;
    return nullptr;
}

void PjrtPluginLibrary::checkError(PJRT_Error* error, const string& operation) const {
    if (error == nullptr)
        return;
    PjrtErrorPtr errorGuard(error, makeErrorDeleter(api));
    string message = getErrorMessageAndDestroy(api, errorGuard.release());
    throw runtime_error("JaxForce PJRT: " + operation + " failed: " + message);
}
