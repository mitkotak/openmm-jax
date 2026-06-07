#include "OpenMmPjrtOutputLifetime.h"
#include <stdexcept>
#include <utility>

using namespace JaxPlugin;
using namespace std;

OpenMmPjrtOutputLifetime::~OpenMmPjrtOutputLifetime() {
    releaseForceOutputsPendingOpenMMUse();
}

void OpenMmPjrtOutputLifetime::reset() {
    destroyForceOutputsPendingOpenMMUse(true);
}

void OpenMmPjrtOutputLifetime::cleanupBeforeExecution() {
    destroyForceOutputsPendingOpenMMUse(false);
}

void OpenMmPjrtOutputLifetime::deferForceOutputDestroyUntilOpenMMUseCompletes(
        PjrtBufferPtr& forceBuffer, CUstream openmmStream) {
    if (forceBuffer == nullptr)
        return;
    forceOutputsPendingOpenMMUse.reserve(forceOutputsPendingOpenMMUse.size()+1);
    RecordedCudaEvent openmmUseComplete = RecordedCudaEvent::record(openmmStream,
            "JaxForce PJRT: failed to create CUDA event for output lifetime tracking",
            "JaxForce PJRT: failed to record CUDA event for output lifetime tracking");
    forceOutputsPendingOpenMMUse.push_back({std::move(forceBuffer), std::move(openmmUseComplete)});
}

void OpenMmPjrtOutputLifetime::destroyForceOutputsPendingOpenMMUse(bool wait) {
    if (wait) {
        waitForOpenMMUseAndDestroyForceOutputs();
        return;
    }
    destroyForceOutputsWhoseOpenMMUseCompleted();
}

void OpenMmPjrtOutputLifetime::destroyForceOutputsWhoseOpenMMUseCompleted() {
    for (size_t i = 0; i < forceOutputsPendingOpenMMUse.size();) {
        ForceOutputPendingOpenMMUse& item = forceOutputsPendingOpenMMUse[i];
        CUresult result = item.openmmUseComplete.query();
        if (result == CUDA_SUCCESS) {
            forceOutputsPendingOpenMMUse.erase(forceOutputsPendingOpenMMUse.begin()+i);
            continue;
        }
        if (result == CUDA_ERROR_NOT_READY) {
            ++i;
            continue;
        }
        throw runtime_error("JaxForce PJRT: failed while checking output pending OpenMM use");
    }
}

void OpenMmPjrtOutputLifetime::waitForOpenMMUseAndDestroyForceOutputs() {
    bool failed = false;
    for (size_t i = 0; i < forceOutputsPendingOpenMMUse.size();) {
        ForceOutputPendingOpenMMUse& item = forceOutputsPendingOpenMMUse[i];
        try {
            item.openmmUseComplete.synchronize(
                    "JaxForce PJRT: failed while waiting for output pending OpenMM use");
            forceOutputsPendingOpenMMUse.erase(forceOutputsPendingOpenMMUse.begin()+i);
        }
        catch (...) {
            failed = true;
            ++i;
        }
    }
    if (failed)
        throw runtime_error("JaxForce PJRT: failed while waiting for output pending OpenMM use");
}

void OpenMmPjrtOutputLifetime::releaseForceOutputsPendingOpenMMUse() noexcept {
    for (auto& item : forceOutputsPendingOpenMMUse)
        item.pjrtBuffer.release();
    forceOutputsPendingOpenMMUse.clear();
}

void OpenMmPjrtOutputLifetime::consumeForceOutput(PjrtBufferPtr forceBuffer,
        CUdeviceptr forcePointer, CUstream openmmStream,
        const function<void(CUdeviceptr)>& consumer) {
    if (forceBuffer == nullptr || forcePointer == 0)
        throw runtime_error("JaxForce PJRT: force output is not available");
    try {
        consumer(forcePointer);
        deferForceOutputDestroyUntilOpenMMUseCompletes(forceBuffer, openmmStream);
        destroyForceOutputsPendingOpenMMUse(false);
    }
    catch (...) {
        if (forceBuffer != nullptr) {
            CUresult syncResult = cuStreamSynchronize(openmmStream);
            if (syncResult != CUDA_SUCCESS) {
                // Do not destroy a PJRT output while OpenMM may still be
                // reading from it and stream completion could not be proven.
                forceBuffer.release();
            }
        }
        destroyForceOutputsPendingOpenMMUse(false);
        throw;
    }
}
