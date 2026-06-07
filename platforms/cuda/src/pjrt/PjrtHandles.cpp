#include "PjrtHandles.h"
#include <stdexcept>

using namespace JaxPlugin;
using namespace std;

string JaxPlugin::getErrorMessageAndDestroy(
        const PJRT_Api* api, PJRT_Error* error) {
    if (error == nullptr)
        return "";
    string message = "unknown error";
    if (api != nullptr && api->PJRT_Error_Message != nullptr) {
        PJRT_Error_Message_Args messageArgs;
        messageArgs.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
        messageArgs.extension_start = nullptr;
        messageArgs.error = error;
        api->PJRT_Error_Message(&messageArgs);
        if (messageArgs.message != nullptr)
            message.assign(messageArgs.message, messageArgs.message_size);
    }
    if (api != nullptr && api->PJRT_Error_Destroy != nullptr) {
        PJRT_Error_Destroy_Args destroyArgs;
        destroyArgs.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        destroyArgs.extension_start = nullptr;
        destroyArgs.error = error;
        api->PJRT_Error_Destroy(&destroyArgs);
    }
    return message;
}


void PjrtBufferDeleter::operator()(PJRT_Buffer* buffer) const {
    if (!buffer)
        return;
    if (api == nullptr || api->PJRT_Buffer_Destroy == nullptr)
        return;
    PJRT_Buffer_Destroy_Args args;
    args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = buffer;
    PJRT_Error* error = api->PJRT_Buffer_Destroy(&args);
    getErrorMessageAndDestroy(api, error);
}

void PjrtEventDeleter::operator()(PJRT_Event* event) const {
    if (!event)
        return;
    if (api == nullptr || api->PJRT_Event_Destroy == nullptr)
        return;
    PJRT_Event_Destroy_Args args;
    args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.event = event;
    PJRT_Error* error = api->PJRT_Event_Destroy(&args);
    getErrorMessageAndDestroy(api, error);
}

void PjrtLoadedExecutableDeleter::operator()(PJRT_LoadedExecutable* executable) const {
    if (!executable)
        return;
    if (api == nullptr || api->PJRT_LoadedExecutable_Destroy == nullptr)
        return;
    PJRT_LoadedExecutable_Destroy_Args args;
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.executable = executable;
    PJRT_Error* error = api->PJRT_LoadedExecutable_Destroy(&args);
    getErrorMessageAndDestroy(api, error);
}

void PjrtClientDeleter::operator()(PJRT_Client* client) const {
    if (!client)
        return;
    if (api == nullptr || api->PJRT_Client_Destroy == nullptr)
        return;
    PJRT_Client_Destroy_Args args;
    args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.client = client;
    PJRT_Error* error = api->PJRT_Client_Destroy(&args);
    getErrorMessageAndDestroy(api, error);
}

void PjrtErrorDeleter::operator()(PJRT_Error* error) const {
    if (!error)
        return;
    if (api == nullptr || api->PJRT_Error_Destroy == nullptr)
        return;
    PJRT_Error_Destroy_Args args;
    args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.error = error;
    api->PJRT_Error_Destroy(&args);
}

PjrtBufferDeleter JaxPlugin::makeBufferDeleter(const PJRT_Api* api) {
    return {api};
}

PjrtEventDeleter JaxPlugin::makeEventDeleter(const PJRT_Api* api) {
    return {api};
}

PjrtLoadedExecutableDeleter JaxPlugin::makeLoadedExecutableDeleter(const PJRT_Api* api) {
    return {api};
}

PjrtClientDeleter JaxPlugin::makeClientDeleter(const PJRT_Api* api) {
    return {api};
}

PjrtErrorDeleter JaxPlugin::makeErrorDeleter(const PJRT_Api* api) {
    return {api};
}
