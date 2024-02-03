#pragma once

#include "../util.cuh"
#include "RefScalarArray.cuh"

namespace feta2 {
namespace core {
/**
 * @brief Array of scalar values with a corresponding copy on the GPU.
 *
 * This class manages two arrays: one on the host and one on the GPU.
 *
 * The host memory is managed RAII-style: allocation happens in the
 * constructor, and deallocation in the destructor.
 *
 * Device memory is allocated when first calling `memcpyHostToDevice` from the
 * host, and deallocated in the destructor.
 *
 * This class is only meant for use on the host. When passing a `ScalarArray`
 * into a `__global__` kernel, use `ScalarArray::deviceRef()` to obtain a
 * non-owning reference to the device array.
 */
template<typename S>
class ScalarArray {
public:
    /** @brief Generic work or global reference type */
    template<ref::RefKind kind>
    using Ref = detail::RefScalarArray<S, kind>;
    /** @brief Global reference type */
    using GRef = Ref<ref::GLOBAL>;
    /** @brief Work reference type */
    using WRef = Ref<ref::WORK>;

    /**
     * @brief Host array construction with element inizialisation.
     *
     * Host memory is allocated and will be deallocated in destructor.
     */
    ScalarArray(const idx_t size, const S& initVal = 0);

    /** @brief Move constructor. */
    ScalarArray(ScalarArray<S>&& other);

    /** @brief Move assignment operator. */
    ScalarArray& operator=(ScalarArray<S>&& other);

    /** @brief Copy construction is disallowed. Use move semantics instead. */
    ScalarArray(ScalarArray<S>&) = delete;
    /** @brief Copy assignment is disallowed. Use move semantics instead. */
    ScalarArray& operator=(ScalarArray<S>&) = delete;

    /** @brief Destructor. Deallocates host and device data as needed. */
    ~ScalarArray();

    /** @brief Async copy data from host to device.
     *
     * Allocates device array if not already allocated.
     */
    void asyncMemcpyHostToDevice(const cudaStream_t stream = 0);

    /** @brief Async copy data from device to host.
     *
     * @throws std::runtime_error if device array is not yet allocated.
     */
    void asyncMemcpyDeviceToHost(const cudaStream_t stream = 0);

    /** @brief Element access on host array. Uses global index.
     *
     * Accesses are bounds-checked if `FETA2_DEBUG_MODE` is defined.
     */
    S& operator[](const SampleIndex& idx) const;

    /** @brief Return pointer to host data. Is never nullptr. */
    inline S* getHostData() const;

    /**
     * @brief Return pointer to device data.
     *
     * @throws if called before calling `memcpyHostToDevice` the first time.
     */
    inline S* getDeviceData() const;

    /** @brief Returns the number of elements in this array. */
    inline idx_t size() const { return size_; }

    /** @brief Return a RefScalarArray pointing to the host array. */
    GRef hostRef() const;

    // /** @brief Return a RefScalarArray pointing to the device array. */
    GRef deviceRef() const;

private:
    /** @brief Allocate device memory */
    void mallocDevice_();

    /** @brief Deallocate data */
    void free_();

    bool hostAllocated_() const { return hostData_ != nullptr; }
    bool deviceAllocated_() const { return deviceData_ != nullptr; }

private:
    S* hostData_   = nullptr;
    S* deviceData_ = nullptr;
    idx_t size_    = 0;
};


template<typename S>
ScalarArray<S>::ScalarArray(ScalarArray<S>&& other)
{
    // Defer logic to move assignment operator
    *this = std::move(other);
}

template<typename S>
ScalarArray<S>& ScalarArray<S>::operator=(ScalarArray<S>&& other)
{
    // Deallocate current arrays, if any
    free_();

    // Take over other's arrays
    hostData_   = other.hostData_;
    size_       = other.size_;
    deviceData_ = other.deviceData_;

    // Prevent other from deallocating data
    other.hostData_   = nullptr;
    other.deviceData_ = nullptr;
    return *this;
}

template<typename S>
ScalarArray<S>::ScalarArray(const idx_t size, const S& initVal)
    : size_{ size }
{
    FETA2_CUAPI(cudaMallocHost(&hostData_, size_ * sizeof(S)));
    std::fill(hostData_, hostData_ + size_, initVal);
}

template<typename S>
void ScalarArray<S>::mallocDevice_()
{
    if (!deviceAllocated_())
        FETA2_CUAPI(cudaMalloc(&deviceData_, size_ * sizeof(S)));
}

template<typename S>
void ScalarArray<S>::free_()
{
    if (hostAllocated_())
        FETA2_CUAPI(cudaFreeHost(hostData_));
    if (deviceAllocated_())
        FETA2_CUAPI(cudaFree(deviceData_));
}

template<typename S>
ScalarArray<S>::~ScalarArray()
{
    free_();
}

template<typename S>
S& ScalarArray<S>::operator[](const SampleIndex& idx) const
{
    return hostRef()[idx.global()];
}

template<typename S>
inline S* ScalarArray<S>::getHostData() const
{
    FETA2_ASSERT(hostAllocated_(), "Host data not allocated!");
    return hostData_;
}

template<typename S>
inline S* ScalarArray<S>::getDeviceData() const
{
    FETA2_ASSERT(deviceAllocated_(), "Device data not allocated!");
    return deviceData_;
}


template<typename S>
void ScalarArray<S>::asyncMemcpyHostToDevice(const cudaStream_t stream)
{
    FETA2_ASSERT(hostAllocated_(), "Host data not allocated!");
    mallocDevice_();
    FETA2_CUAPI(cudaMemcpyAsync(getDeviceData(), getHostData(),
        size() * sizeof(S), cudaMemcpyHostToDevice, stream));
}

template<typename S>
void ScalarArray<S>::asyncMemcpyDeviceToHost(const cudaStream_t stream)
{
    FETA2_CUAPI(cudaMemcpyAsync(getHostData(), getDeviceData(),
        size() * sizeof(S), cudaMemcpyDeviceToHost, stream));
}

template<typename S>
typename ScalarArray<S>::GRef ScalarArray<S>::deviceRef() const
{
    FETA2_ASSERT(deviceAllocated_(),
        "`deviceRef` called without initialization of device memory! Call "
        "`asyncMemcpyHostToDevice` first.");
    return GRef(deviceData_, size_);
}

template<typename S>
typename ScalarArray<S>::GRef ScalarArray<S>::hostRef() const
{
    FETA2_ASSERT(hostAllocated_(),
        "ScalarArray::hostRef called without initialization!");
    return GRef(hostData_, size_);
}

} // namespace core
} // namespace feta2
