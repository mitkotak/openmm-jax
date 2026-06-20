/* CUDA PJRT stream extension used by jax-cuda PJRT for external buffer handoff.

   The core PJRT C API header is vendored directly from the XLA revision used by
   jaxlib 0.10.1. This extension is not declared in that public header, so keep
   the local declaration separate from the vendored file.

   Upstream References:
   - C++ counterpart method: `xla::PjRtDevice::GetStreamForExternalReadyEvents` in `xla/pjrt/pjrt_client.h`
   - C-API enum value: `PJRT_Extension_Type_Stream` in `xla/pjrt/c/pjrt_c_api.h` */

#ifndef OPENMM_PJRT_CUDA_STREAM_EXTENSION_H_
#define OPENMM_PJRT_CUDA_STREAM_EXTENSION_H_

#include "pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PJRT_Get_Stream_For_External_Ready_Events_Args {
    size_t struct_size;
    PJRT_Device* device;
    intptr_t stream;
} PJRT_Get_Stream_For_External_Ready_Events_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Get_Stream_For_External_Ready_Events_Args, stream);

typedef PJRT_Error* PJRT_Get_Stream_For_External_Ready_Events(
        PJRT_Get_Stream_For_External_Ready_Events_Args* args);

typedef struct PJRT_Stream_Extension {
    PJRT_Extension_Base base;
    PJRT_Get_Stream_For_External_Ready_Events* get_stream;
} PJRT_Stream_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Stream_Extension, get_stream);

#ifdef __cplusplus
}
#endif

#endif
