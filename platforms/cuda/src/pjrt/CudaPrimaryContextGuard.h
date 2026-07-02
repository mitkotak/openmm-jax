#ifndef OPENMM_CUDA_PRIMARY_CONTEXT_GUARD_H_
#define OPENMM_CUDA_PRIMARY_CONTEXT_GUARD_H_

#include "openmm/OpenMMException.h"
#include "openmm/cuda/CudaContext.h"
#include <cuda.h>
#include <sstream>
#include <string>

namespace JaxPlugin {

/**
 * RAII guard that retains the CUDA primary context for the device used
 * by an OpenMM CudaContext.  Releases the primary context on destruction.
 */
class PrimaryContextRetainer {
public:
    explicit PrimaryContextRetainer(OpenMM::CudaContext& cu)
            : cu(cu), context(nullptr), active(false) {
        check(cuDevicePrimaryCtxRetain(&context, cu.getDevice()),
                "Failed to retain the CUDA primary context");
        active = true;
    }

    PrimaryContextRetainer(const PrimaryContextRetainer&) = delete;
    PrimaryContextRetainer& operator=(const PrimaryContextRetainer&) = delete;

    ~PrimaryContextRetainer() {
        if (active)
            cuDevicePrimaryCtxRelease(cu.getDevice());
    }

    CUcontext get() const {
        return context;
    }

private:
    void check(CUresult result, const std::string& prefix) {
        if (result != CUDA_SUCCESS) {
            std::stringstream m;
            m << prefix << ": " << cu.getErrorString(result)
              << " (" << result << ")";
            throw OpenMM::OpenMMException(m.str());
        }
    }

    OpenMM::CudaContext& cu;
    CUcontext context;
    bool active;
};

/**
 * RAII guard that makes the CUDA primary context current for PJRT operations.
 */
class ScopedPrimaryContext {
public:
    ScopedPrimaryContext(OpenMM::CudaContext& cu, CUcontext context)
            : cu(cu), expectedContext(context), previousContext(nullptr),
              active(true) {
        check(cuCtxGetCurrent(&previousContext),
                "Failed to get the current CUDA context before PJRT");
        check(cuCtxSetCurrent(expectedContext),
                "Failed to set the CUDA primary context for PJRT");
    }

    ScopedPrimaryContext(const ScopedPrimaryContext&) = delete;
    ScopedPrimaryContext& operator=(const ScopedPrimaryContext&) = delete;

    ~ScopedPrimaryContext() {
        if (active) {
            try {
                restore(false);
            } catch (...) {
            }
        }
    }

    void restore() {
        restore(true);
    }

private:
    void restore(bool throwOnError) {
        CUcontext current;
        CUresult result = cuCtxGetCurrent(&current);
        if (result != CUDA_SUCCESS) {
            if (throwOnError)
                check(result, "Failed to get the current CUDA context after PJRT");
            return;
        }
        if (current != expectedContext) {
            if (throwOnError)
                throw OpenMM::OpenMMException(
                        "JaxForce CUDA backend found an unexpected CUDA context after PJRT");
            return;
        }
        result = cuCtxSetCurrent(previousContext);
        if (result != CUDA_SUCCESS) {
            if (throwOnError)
                check(result, "Failed to restore the CUDA context after PJRT");
            return;
        }
        active = false;
    }

    void check(CUresult result, const std::string& prefix) {
        if (result != CUDA_SUCCESS) {
            std::stringstream m;
            m << prefix << ": " << cu.getErrorString(result)
              << " (" << result << ")";
            throw OpenMM::OpenMMException(m.str());
        }
    }

    OpenMM::CudaContext& cu;
    CUcontext expectedContext;
    CUcontext previousContext;
    bool active;
};

} // namespace JaxPlugin

#endif
