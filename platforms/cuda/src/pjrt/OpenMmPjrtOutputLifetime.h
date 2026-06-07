#ifndef OPENMM_OPENMM_PJRT_OUTPUT_LIFETIME_H_
#define OPENMM_OPENMM_PJRT_OUTPUT_LIFETIME_H_

#include "CudaEvent.h"
#include "PjrtHandles.h"
#include <cuda.h>
#include <functional>
#include <vector>

namespace JaxPlugin {


class OpenMmPjrtOutputLifetime {
public:
    ~OpenMmPjrtOutputLifetime();

    void reset();

    void cleanupBeforeExecution();

    void consumeForceOutput(PjrtBufferPtr forceBuffer, CUdeviceptr forcePointer,
            CUstream openmmStream,
            const std::function<void(CUdeviceptr)>& consumer);

private:
    struct ForceOutputPendingOpenMMUse {
        PjrtBufferPtr pjrtBuffer;
        RecordedCudaEvent openmmUseComplete;
    };

    void deferForceOutputDestroyUntilOpenMMUseCompletes(PjrtBufferPtr& forceBuffer,
            CUstream openmmStream);
    void destroyForceOutputsPendingOpenMMUse(bool wait);
    void destroyForceOutputsWhoseOpenMMUseCompleted();
    void waitForOpenMMUseAndDestroyForceOutputs();
    void releaseForceOutputsPendingOpenMMUse() noexcept;

    std::vector<ForceOutputPendingOpenMMUse> forceOutputsPendingOpenMMUse;
};

} // namespace JaxPlugin

#endif
