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
 * RAII guard that pushes the CUDA primary context for PJRT operations and
 * restores the OpenMM context on destruction or explicit pop().
 */
class ScopedPrimaryContext {
public:
    ScopedPrimaryContext(OpenMM::CudaContext& cu, CUcontext context)
            : cu(cu), expectedContext(context), active(true) {
        check(cuCtxPushCurrent(expectedContext),
                "Failed to push the CUDA primary context for PJRT");
    }

    ScopedPrimaryContext(const ScopedPrimaryContext&) = delete;
    ScopedPrimaryContext& operator=(const ScopedPrimaryContext&) = delete;

    ~ScopedPrimaryContext() {
        if (active) {
            cuCtxPopCurrent(nullptr);
            cu.setAsCurrent();
        }
    }

    /** Explicitly pop the primary context and restore OpenMM's context. */
    void pop() {
        CUcontext popped;
        check(cuCtxPopCurrent(&popped),
                "Failed to pop the CUDA primary context after PJRT");
        active = false;
        if (popped != expectedContext)
            throw OpenMM::OpenMMException(
                    "JaxForce CUDA backend popped an unexpected CUDA context");
        cu.setAsCurrent();
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
    CUcontext expectedContext;
    bool active;
};

} // namespace JaxPlugin

#endif

