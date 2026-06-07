#ifndef OPENMM_CUDA_EVENT_H_
#define OPENMM_CUDA_EVENT_H_

#include <cuda.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace JaxPlugin {

class RecordedCudaEvent {
public:
    RecordedCudaEvent() : event(nullptr) {
    }
    RecordedCudaEvent(const RecordedCudaEvent&) = delete;
    RecordedCudaEvent& operator=(const RecordedCudaEvent&) = delete;
    RecordedCudaEvent(RecordedCudaEvent&& other) noexcept : event(other.event) {
        other.event = nullptr;
    }
    RecordedCudaEvent& operator=(RecordedCudaEvent&& other) noexcept {
        if (this != &other) {
            reset();
            event = other.event;
            other.event = nullptr;
        }
        return *this;
    }
    ~RecordedCudaEvent() {
        reset();
    }

    static RecordedCudaEvent record(CUstream stream, const std::string& createError,
            const std::string& recordError) {
        RecordedCudaEvent recorded;
        CUresult result = cuEventCreate(&recorded.event, CU_EVENT_DISABLE_TIMING);
        if (result != CUDA_SUCCESS)
            throw std::runtime_error(createError);
        result = cuEventRecord(recorded.event, stream);
        if (result != CUDA_SUCCESS) {
            recorded.reset();
            throw std::runtime_error(recordError);
        }
        return recorded;
    }

    CUevent get() const {
        return event;
    }

    /** Non-blocking query: returns CUDA_SUCCESS or CUDA_ERROR_NOT_READY. */
    CUresult query() const {
        return cuEventQuery(event);
    }

    /** Blocking wait for the event to complete. */
    void synchronize(const std::string& error) const {
        CUresult result = cuEventSynchronize(event);
        if (result != CUDA_SUCCESS)
            throw std::runtime_error(error);
    }

private:
    void reset() {
        if (event != nullptr)
            cuEventDestroy(event);
        event = nullptr;
    }

    CUevent event;
};

} // namespace JaxPlugin

#endif

