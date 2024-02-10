#pragma once

#include "../util.cuh"
#include "RefScalarEnsemble.cuh"

namespace feta2 {
namespace core {
/**
 * @brief Ensemble of scalar values with a corresponding copy on the GPU.
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
template<typename Scalar_>
class ScalarEnsemble {
public:
    /** @brief Generic work or global reference type */
    template<ref::RefKind kind>
    using Ref = detail::RefScalarEnsemble<Scalar_, kind>;
    /** @brief Global reference type */
    using GRef = Ref<ref::GLOBAL>;
    /** @brief Work reference type */
    using WRef = Ref<ref::WORK>;

    /**
     * @brief Host array construction with element inizialisation.
     *
     * Host memory is allocated and will be deallocated in destructor.
     */
    ScalarEnsemble(const idx_t size, const Scalar_& initVal = 0);

    /** @brief Move constructor. */
    ScalarEnsemble(ScalarEnsemble<Scalar_>&& other);

    /** @brief Move assignment operator. */
    ScalarEnsemble& operator=(ScalarEnsemble<Scalar_>&& other);

    /** @brief Copy construction is disallowed. Use move semantics instead. */
    ScalarEnsemble(ScalarEnsemble<Scalar_>&) = delete;
    /** @brief Copy assignment is disallowed. Use move semantics instead. */
    ScalarEnsemble& operator=(ScalarEnsemble<Scalar_>&) = delete;

    /** @brief Destructor. Deallocates host and device data as needed. */
    ~ScalarEnsemble();

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
    Scalar_& operator[](const SampleIndex& idx) const;

    /** @brief Element access on host array.
     *
     * Accesses are bounds-checked if `FETA2_DEBUG_MODE` is defined.
     */
    Scalar_& operator[](const idx_t& idx) const;

    /** @brief Return pointer to host data. Is never nullptr. */
    inline Scalar_* hostData() const;

    /**
     * @brief Return pointer to device data.
     *
     * @throws if called before calling `memcpyHostToDevice` the first time.
     */
    inline Scalar_* deviceData() const;

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
    Scalar_* hostData_   = nullptr;
    Scalar_* deviceData_ = nullptr;
    idx_t size_          = 0;
};


template<typename S>
ScalarEnsemble<S>::ScalarEnsemble(ScalarEnsemble<S>&& other)
{
    // Defer logic to move assignment operator
    *this = std::move(other);
}

template<typename S>
ScalarEnsemble<S>& ScalarEnsemble<S>::operator=(ScalarEnsemble<S>&& other)
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
ScalarEnsemble<S>::ScalarEnsemble(const idx_t size, const S& initVal)
    : size_{ size }
{
    FETA2_CUAPI(cudaMallocHost(&hostData_, size_ * sizeof(S)));
    std::fill(hostData_, hostData_ + size_, initVal);
}

template<typename S>
void ScalarEnsemble<S>::mallocDevice_()
{
    if (!deviceAllocated_())
        FETA2_CUAPI(cudaMalloc(&deviceData_, size_ * sizeof(S)));
}

template<typename S>
void ScalarEnsemble<S>::free_()
{
    if (hostAllocated_())
        FETA2_CUAPI(cudaFreeHost(hostData_));
    if (deviceAllocated_())
        FETA2_CUAPI(cudaFree(deviceData_));
}

template<typename S>
ScalarEnsemble<S>::~ScalarEnsemble()
{
    free_();
}

template<typename S>
S& ScalarEnsemble<S>::operator[](const SampleIndex& idx) const
{
    return hostRef()[idx.global()];
}

template<typename S>
S& ScalarEnsemble<S>::operator[](const idx_t& idx) const
{
    return hostRef()[idx];
}

template<typename S>
inline S* ScalarEnsemble<S>::hostData() const
{
    FETA2_ASSERT(hostAllocated_(), "Host data not allocated!");
    return hostData_;
}

template<typename S>
inline S* ScalarEnsemble<S>::deviceData() const
{
    FETA2_ASSERT(deviceAllocated_(), "Device data not allocated!");
    return deviceData_;
}


template<typename S>
void ScalarEnsemble<S>::asyncMemcpyHostToDevice(const cudaStream_t stream)
{
    FETA2_ASSERT(hostAllocated_(), "Host data not allocated!");
    mallocDevice_();
    FETA2_CUAPI(cudaMemcpyAsync(deviceData(), hostData(), size() * sizeof(S),
        cudaMemcpyHostToDevice, stream));
}

template<typename S>
void ScalarEnsemble<S>::asyncMemcpyDeviceToHost(const cudaStream_t stream)
{
    FETA2_CUAPI(cudaMemcpyAsync(hostData(), deviceData(), size() * sizeof(S),
        cudaMemcpyDeviceToHost, stream));
}

template<typename S>
typename ScalarEnsemble<S>::GRef ScalarEnsemble<S>::deviceRef() const
{
    FETA2_ASSERT(deviceAllocated_(),
        "`deviceRef` called without initialization of device memory! Call "
        "`asyncMemcpyHostToDevice` first.");
    return GRef(deviceData_, size_);
}

template<typename S>
typename ScalarEnsemble<S>::GRef ScalarEnsemble<S>::hostRef() const
{
    FETA2_ASSERT(hostAllocated_(),
        "ScalarArray::hostRef called without initialization!");
    return GRef(hostData_, size_);
}

} // namespace core
} // namespace feta2
