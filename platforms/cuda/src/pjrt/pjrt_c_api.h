/* Copyright 2022 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PJRT_C_PJRT_C_API_H_
#define XLA_PJRT_C_PJRT_C_API_H_

/* Pruned local copy of XLA's PJRT C API header, kept as a header-only
   dependency for loading external PJRT plugins without linking XLA. */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define PJRT_STRUCT_SIZE(struct_type, last_field) \
    (offsetof(struct_type, last_field) + sizeof(((struct_type*)0)->last_field))

#ifdef __cplusplus
#define PJRT_CHECK_STRUCT_SIZE(sname, last_field)                       \
  static_assert(                                                        \
      sizeof(struct sname) ==                                           \
          ((PJRT_STRUCT_SIZE(sname, last_field) + alignof(struct sname) - 1) / \
           alignof(struct sname)) *                                            \
              alignof(struct sname),                                           \
      "Failed to update last_field");
#else
#define PJRT_CHECK_STRUCT_SIZE(sname, last_field)
#endif

#define PJRT_DEFINE_STRUCT_TRAITS(sname, last_field)                  \
  typedef struct sname sname;                                         \
  enum { sname##_STRUCT_SIZE = PJRT_STRUCT_SIZE(sname, last_field) }; \
  PJRT_CHECK_STRUCT_SIZE(sname, last_field)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PJRT_Extension_Type_Stream = 3,
} PJRT_Extension_Type;

typedef struct PJRT_Extension_Base {
    size_t struct_size;
    PJRT_Extension_Type type;
    struct PJRT_Extension_Base* next;
} PJRT_Extension_Base;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Extension_Base, next);

typedef struct PJRT_Api_Version {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    int major_version;
    int minor_version;
} PJRT_Api_Version;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Api_Version, minor_version);

typedef struct PJRT_Error PJRT_Error;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_Buffer PJRT_Buffer;

typedef struct PJRT_Get_Stream_For_External_Ready_Events_Args {
    size_t struct_size;
    PJRT_Device* device;
    intptr_t stream;
} PJRT_Get_Stream_For_External_Ready_Events_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Get_Stream_For_External_Ready_Events_Args, stream);

typedef PJRT_Error* PJRT_Get_Stream_For_External_Ready_Events_Fn(PJRT_Get_Stream_For_External_Ready_Events_Args* args);

typedef struct PJRT_Stream_Extension {
    PJRT_Extension_Base base;
    PJRT_Get_Stream_For_External_Ready_Events_Fn* get_stream;
} PJRT_Stream_Extension;

typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Memory PJRT_Memory;
typedef struct PJRT_ExecuteContext PJRT_ExecuteContext;
typedef struct PJRT_MultiSlice_Config PJRT_MultiSlice_Config;

typedef struct PJRT_Error_Destroy_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_Error* error;
} PJRT_Error_Destroy_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Error_Destroy_Args, error);

typedef struct PJRT_Error_Message_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    const PJRT_Error* error;
    const char* message;
    size_t message_size;
} PJRT_Error_Message_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Error_Message_Args, message_size);

typedef void PJRT_Error_Destroy_Fn(PJRT_Error_Destroy_Args* args);
typedef void PJRT_Error_Message_Fn(PJRT_Error_Message_Args* args);

typedef enum {
    PJRT_NamedValue_kString = 0,
    PJRT_NamedValue_kInt64,
    PJRT_NamedValue_kInt64List,
    PJRT_NamedValue_kFloat,
    PJRT_NamedValue_kBool,
} PJRT_NamedValue_Type;

typedef struct PJRT_NamedValue {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    const char* name;
    size_t name_size;
    PJRT_NamedValue_Type type;
    union {
        const char* string_value;
        int64_t int64_value;
        const int64_t* int64_array_value;
        float float_value;
        bool bool_value;
    };
    size_t value_size;
} PJRT_NamedValue;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_NamedValue, value_size);

typedef struct PJRT_Plugin_Initialize_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
} PJRT_Plugin_Initialize_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Plugin_Initialize_Args, extension_start);

typedef PJRT_Error* PJRT_Plugin_Initialize_Fn(PJRT_Plugin_Initialize_Args* args);

typedef struct PJRT_Event PJRT_Event;
typedef struct PJRT_Event_Destroy_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_Event* event;
} PJRT_Event_Destroy_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_Destroy_Args, event);

typedef struct PJRT_Event_Await_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_Event* event;
} PJRT_Event_Await_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_Await_Args, event);

typedef PJRT_Error* PJRT_Event_Destroy_Fn(PJRT_Event_Destroy_Args* args);
typedef PJRT_Error* PJRT_Event_Await_Fn(PJRT_Event_Await_Args* args);

typedef struct PJRT_Client_Create_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    const PJRT_NamedValue* create_options;
    size_t num_options;
    void* kv_get_callback;
    void* kv_get_user_arg;
    void* kv_put_callback;
    void* kv_put_user_arg;
    PJRT_Client* client;
    void* kv_try_get_callback;
    void* kv_try_get_user_arg;
} PJRT_Client_Create_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Create_Args, kv_try_get_user_arg);

typedef struct PJRT_Client_Destroy_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_Client* client;
} PJRT_Client_Destroy_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Destroy_Args, client);

typedef struct PJRT_Client_Devices_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_Client* client;
    PJRT_Device* const* devices;
    size_t num_devices;
} PJRT_Client_Devices_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Devices_Args, num_devices);

typedef struct PJRT_Program {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    char* code;
    size_t code_size;
    const char* format;
    size_t format_size;
} PJRT_Program;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Program, format_size);

typedef struct PJRT_Client_Compile_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_Client* client;
    const PJRT_Program* program;
    const char* compile_options;
    size_t compile_options_size;
    PJRT_LoadedExecutable* executable;
} PJRT_Client_Compile_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Compile_Args, executable);

typedef struct PJRT_LoadedExecutable_Destroy_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_LoadedExecutable* executable;
} PJRT_LoadedExecutable_Destroy_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_Destroy_Args, executable);

typedef enum {
    PJRT_Buffer_Type_F32 = 11,
} PJRT_Buffer_Type;

typedef enum {
    PJRT_Buffer_MemoryLayout_Type_Tiled = 0,
} PJRT_Buffer_MemoryLayout_Type;

typedef struct PJRT_Buffer_MemoryLayout_Tiled {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    const int64_t* minor_to_major;
    size_t minor_to_major_size;
    const int64_t* tile_dims;
    const size_t* tile_dim_sizes;
    size_t num_tiles;
} PJRT_Buffer_MemoryLayout_Tiled;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_MemoryLayout_Tiled, num_tiles);

typedef struct PJRT_Buffer_MemoryLayout {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_Buffer_MemoryLayout_Tiled tiled;
    PJRT_Buffer_MemoryLayout_Type type;
} PJRT_Buffer_MemoryLayout;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_MemoryLayout, type);

typedef struct PJRT_Client_CreateViewOfDeviceBuffer_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_Client* client;
    void* device_buffer_ptr;
    const int64_t* dims;
    size_t num_dims;
    PJRT_Buffer_Type element_type;
    PJRT_Buffer_MemoryLayout* layout;
    PJRT_Device* device;
    void (*on_delete_callback)(void* device_buffer_ptr, void* user_arg);
    void* on_delete_callback_arg;
    intptr_t stream;
    PJRT_Buffer* buffer;
    PJRT_Memory* memory;
} PJRT_Client_CreateViewOfDeviceBuffer_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_CreateViewOfDeviceBuffer_Args, memory);

typedef struct PJRT_ExecuteOptions {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    void* send_callbacks;
    void* recv_callbacks;
    size_t num_send_ops;
    size_t num_recv_ops;
    int launch_id;
    const int64_t* non_donatable_input_indices;
    size_t num_non_donatable_input_indices;
    PJRT_ExecuteContext* context;
    const char* call_location;
    size_t num_tasks;
    int* task_ids;
    int64_t* incarnation_ids;
    PJRT_MultiSlice_Config* multi_slice_config;
} PJRT_ExecuteOptions;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecuteOptions, multi_slice_config);

typedef struct PJRT_LoadedExecutable_Execute_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_LoadedExecutable* executable;
    PJRT_ExecuteOptions* options;
    PJRT_Buffer* const* const* argument_lists;
    size_t num_devices;
    size_t num_args;
    PJRT_Buffer** const* output_lists;
    PJRT_Event** device_complete_events;
    PJRT_Device* execute_device;
} PJRT_LoadedExecutable_Execute_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_Execute_Args, execute_device);

typedef struct PJRT_Buffer_Destroy_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_Buffer* buffer;
} PJRT_Buffer_Destroy_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_Destroy_Args, buffer);

typedef struct PJRT_Buffer_IncreaseExternalReferenceCount_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_Buffer* buffer;
} PJRT_Buffer_IncreaseExternalReferenceCount_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_IncreaseExternalReferenceCount_Args, buffer);

typedef struct PJRT_Buffer_DecreaseExternalReferenceCount_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_Buffer* buffer;
} PJRT_Buffer_DecreaseExternalReferenceCount_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_DecreaseExternalReferenceCount_Args, buffer);

typedef struct PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_Buffer* buffer;
    void* device_memory_ptr;
} PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args, device_memory_ptr);

typedef PJRT_Error* PJRT_Client_Create_Fn(PJRT_Client_Create_Args* args);
typedef PJRT_Error* PJRT_Client_Destroy_Fn(PJRT_Client_Destroy_Args* args);
typedef PJRT_Error* PJRT_Client_Devices_Fn(PJRT_Client_Devices_Args* args);
typedef PJRT_Error* PJRT_Client_Compile_Fn(PJRT_Client_Compile_Args* args);
typedef PJRT_Error* PJRT_LoadedExecutable_Destroy_Fn(PJRT_LoadedExecutable_Destroy_Args* args);
typedef PJRT_Error* PJRT_LoadedExecutable_Execute_Fn(PJRT_LoadedExecutable_Execute_Args* args);
typedef PJRT_Error* PJRT_Buffer_Destroy_Fn(PJRT_Buffer_Destroy_Args* args);
typedef PJRT_Error* PJRT_Buffer_IncreaseExternalReferenceCount_Fn(PJRT_Buffer_IncreaseExternalReferenceCount_Args* args);
typedef PJRT_Error* PJRT_Buffer_DecreaseExternalReferenceCount_Fn(PJRT_Buffer_DecreaseExternalReferenceCount_Args* args);
typedef PJRT_Error* PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Fn(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args);
typedef PJRT_Error* PJRT_Client_CreateViewOfDeviceBuffer_Fn(PJRT_Client_CreateViewOfDeviceBuffer_Args* args);

typedef struct PJRT_Api {
    size_t struct_size;
    PJRT_Extension_Base* extension_start;
    PJRT_Api_Version pjrt_api_version;
    PJRT_Error_Destroy_Fn* PJRT_Error_Destroy;
    PJRT_Error_Message_Fn* PJRT_Error_Message;
    void* PJRT_Error_GetCode;
    PJRT_Plugin_Initialize_Fn* PJRT_Plugin_Initialize;
    void* PJRT_Plugin_Attributes;
    PJRT_Event_Destroy_Fn* PJRT_Event_Destroy;
    void* PJRT_Event_IsReady;
    void* PJRT_Event_Error;
    PJRT_Event_Await_Fn* PJRT_Event_Await;
    void* PJRT_Event_OnReady;
    PJRT_Client_Create_Fn* PJRT_Client_Create;
    PJRT_Client_Destroy_Fn* PJRT_Client_Destroy;
    void* PJRT_Client_PlatformName;
    void* PJRT_Client_ProcessIndex;
    void* PJRT_Client_PlatformVersion;
    PJRT_Client_Devices_Fn* PJRT_Client_Devices;
    void* PJRT_Client_AddressableDevices;
    void* PJRT_Client_LookupDevice;
    void* PJRT_Client_LookupAddressableDevice;
    void* PJRT_Client_AddressableMemories;
    PJRT_Client_Compile_Fn* PJRT_Client_Compile;
    void* PJRT_Client_DefaultDeviceAssignment;
    void* PJRT_Client_BufferFromHostBuffer;
    void* PJRT_DeviceDescription_Id;
    void* PJRT_DeviceDescription_ProcessIndex;
    void* PJRT_DeviceDescription_Attributes;
    void* PJRT_DeviceDescription_Kind;
    void* PJRT_DeviceDescription_DebugString;
    void* PJRT_DeviceDescription_ToString;
    void* PJRT_Device_GetDescription;
    void* PJRT_Device_IsAddressable;
    void* PJRT_Device_LocalHardwareId;
    void* PJRT_Device_AddressableMemories;
    void* PJRT_Device_DefaultMemory;
    void* PJRT_Device_MemoryStats;
    void* PJRT_Memory_Id;
    void* PJRT_Memory_Kind;
    void* PJRT_Memory_DebugString;
    void* PJRT_Memory_ToString;
    void* PJRT_Memory_AddressableByDevices;
    void* PJRT_Executable_Destroy;
    void* PJRT_Executable_Name;
    void* PJRT_Executable_NumReplicas;
    void* PJRT_Executable_NumPartitions;
    void* PJRT_Executable_NumOutputs;
    void* PJRT_Executable_SizeOfGeneratedCodeInBytes;
    void* PJRT_Executable_GetCostAnalysis;
    void* PJRT_Executable_OutputMemoryKinds;
    void* PJRT_Executable_OptimizedProgram;
    void* PJRT_Executable_Serialize;
    PJRT_LoadedExecutable_Destroy_Fn* PJRT_LoadedExecutable_Destroy;
    void* PJRT_LoadedExecutable_GetExecutable;
    void* PJRT_LoadedExecutable_AddressableDevices;
    void* PJRT_LoadedExecutable_Delete;
    void* PJRT_LoadedExecutable_IsDeleted;
    PJRT_LoadedExecutable_Execute_Fn* PJRT_LoadedExecutable_Execute;
    void* PJRT_Executable_DeserializeAndLoad;
    void* PJRT_LoadedExecutable_Fingerprint;
    PJRT_Buffer_Destroy_Fn* PJRT_Buffer_Destroy;
    void* PJRT_Buffer_ElementType;
    void* PJRT_Buffer_Dimensions;
    void* PJRT_Buffer_UnpaddedDimensions;
    void* PJRT_Buffer_DynamicDimensionIndices;
    void* PJRT_Buffer_GetMemoryLayout;
    void* PJRT_Buffer_OnDeviceSizeInBytes;
    void* PJRT_Buffer_Device;
    void* PJRT_Buffer_Memory;
    void* PJRT_Buffer_Delete;
    void* PJRT_Buffer_IsDeleted;
    void* PJRT_Buffer_CopyToDevice;
    void* PJRT_Buffer_ToHostBuffer;
    void* PJRT_Buffer_IsOnCpu;
    void* PJRT_Buffer_ReadyEvent;
    void* PJRT_Buffer_UnsafePointer;
    PJRT_Buffer_IncreaseExternalReferenceCount_Fn* PJRT_Buffer_IncreaseExternalReferenceCount;
    PJRT_Buffer_DecreaseExternalReferenceCount_Fn* PJRT_Buffer_DecreaseExternalReferenceCount;
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Fn* PJRT_Buffer_OpaqueDeviceMemoryDataPointer;
    void* PJRT_CopyToDeviceStream_Destroy;
    void* PJRT_CopyToDeviceStream_AddChunk;
    void* PJRT_CopyToDeviceStream_TotalBytes;
    void* PJRT_CopyToDeviceStream_GranuleSize;
    void* PJRT_CopyToDeviceStream_CurrentBytes;
    void* PJRT_TopologyDescription_Create;
    void* PJRT_TopologyDescription_Destroy;
    void* PJRT_TopologyDescription_PlatformName;
    void* PJRT_TopologyDescription_PlatformVersion;
    void* PJRT_TopologyDescription_GetDeviceDescriptions;
    void* PJRT_TopologyDescription_Serialize;
    void* PJRT_TopologyDescription_Attributes;
    void* PJRT_Compile;
    void* PJRT_Executable_OutputElementTypes;
    void* PJRT_Executable_OutputDimensions;
    void* PJRT_Buffer_CopyToMemory;
    PJRT_Client_CreateViewOfDeviceBuffer_Fn* PJRT_Client_CreateViewOfDeviceBuffer;
} PJRT_Api;

typedef const PJRT_Api* GetPjrtApiFn();

#ifdef __cplusplus
}
#endif

#endif
