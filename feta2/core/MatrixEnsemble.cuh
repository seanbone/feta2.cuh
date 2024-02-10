#pragma once

#include "../util.cuh"
#include "RefMatrixEnsemble.cuh"
#include "ScalarEnsemble.cuh"

namespace feta2 {
namespace core {

// TODO doc
template<typename Scalar_, dim_t Rows_, dim_t Cols_>
class MatrixEnsemble {
    using StrideT = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

public:
    /** @brief Generic work or global reference type */
    template<ref::RefKind kind>
    using Ref = detail::RefMatrixEnsemble<Scalar_, Rows_, Cols_, kind>;
    /** @brief Global reference type */
    using GRef = Ref<ref::GLOBAL>;
    /** @brief Work reference type */
    using WRef = Ref<ref::WORK>;

    /** @brief Equivalent type for one ensemble element */
    using Element = typename GRef::Element;
    /** @brief Mapped type for one ensemble element */
    using ElementMap = typename GRef::ElementMap;

    /** @brief Host construction and initialization.
     *
     * Host memory is allocated, but not device memory.
     */
    MatrixEnsemble(const idx_t& size, const Scalar_& initVal = 0);

    /** @brief Ensemble size, ie number of matrices contained */
    inline idx_t size() const { return size_; }

    /** @brief Element access operator on host array. */
    ElementMap operator[](const idx_t& idx) const;

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

    /** @brief Return a non-owning reference to the host-allocated array. */
    GRef hostRef() const;

    /** @brief Return a non-owning reference to the device-allocated array. */
    GRef deviceRef() const;

    /** @brief Return pointer to host data. Is never nullptr. */
    inline Scalar_* hostData() const { return data_.hostData(); }

    /**
     * @brief Return pointer to device data.
     *
     * @throws if called before calling `memcpyHostToDevice` the first time.
     */
    inline Scalar_* deviceData() const { return data_.deviceData(); }

private:
    ScalarEnsemble<Scalar_> data_;
    idx_t size_;
};

template<typename S, dim_t R, dim_t C>
MatrixEnsemble<S, R, C>::MatrixEnsemble(const idx_t& size, const S& initVal)
    : data_(size * R * C, initVal)
    , size_{ size }
{
}

template<typename S, dim_t R, dim_t C>
typename MatrixEnsemble<S, R, C>::GRef MatrixEnsemble<S, R, C>::hostRef() const
{
    return GRef(data_.hostRef(), size_);
}

template<typename S, dim_t R, dim_t C>
typename MatrixEnsemble<S, R, C>::GRef
MatrixEnsemble<S, R, C>::deviceRef() const
{
    return GRef(data_.deviceRef(), size_);
}

template<typename S, dim_t Rows_, dim_t Cols_>
typename MatrixEnsemble<S, Rows_, Cols_>::ElementMap
MatrixEnsemble<S, Rows_, Cols_>::operator[](const idx_t& idx) const
{
    return hostRef()[idx];
}

template<typename S, dim_t Rows_, dim_t Cols_>
void MatrixEnsemble<S, Rows_, Cols_>::asyncMemcpyHostToDevice(
    const cudaStream_t stream)
{
    data_.asyncMemcpyHostToDevice(stream);
}

template<typename S, dim_t Rows_, dim_t Cols_>
void MatrixEnsemble<S, Rows_, Cols_>::asyncMemcpyDeviceToHost(
    const cudaStream_t stream)
{
    data_.asyncMemcpyDeviceToHost(stream);
}
} // namespace core
} // namespace feta2
