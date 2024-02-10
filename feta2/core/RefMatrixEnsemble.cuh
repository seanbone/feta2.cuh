#pragma once

#include "../util.cuh"
#include "ScalarEnsemble.cuh"
#include "feta2/util/typedefs.cuh"

namespace feta2 {
namespace core {
namespace detail {

// TODO doc
template<typename Scalar_, dim_t Rows_, dim_t Cols_, ref::RefKind kind_>
class RefMatrixEnsemble {
    using ScalarRef_ = typename ScalarEnsemble<Scalar_>::template Ref<kind_>;
    using StrideT    = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

public:
    /** @brief Equivalent type for one ensemble element */
    using Element = Eigen::Matrix<Scalar_, Rows_, Cols_>;
    /** @brief Mapped type for one ensemble element */
    using ElementMap = Eigen::Map<Element, Eigen::Unaligned, StrideT>;

    FETA2_DEVICE RefMatrixEnsemble(ScalarRef_&& data, idx_t size);

    /**
     * @brief Construct a non-owning matrix ensemble assuming `data` has
     * been allocated for at least `bufSize(size)` bytes.
     */
    FETA2_DEVICE RefMatrixEnsemble(Scalar_* const data, const idx_t size);

    /** @brief Number of elements in the referenced ensemble. */
    FETA2_DEVICE idx_t size() const { return size_; }

    /** @brief Element access operator on referenced array. */
    FETA2_DEVICE ElementMap operator[](const idx_t& idx);

    /** @brief Element access operator with automatic global/work indexing.
     *
     * Uses the `SampleIndex`'s global or work index depending on whether `this`
     * is a global or work reference.
     */
    FETA2_DEVICE ElementMap operator[](const SampleIndex& idx);

    /**
     * @brief Returns a work ensemble reference of the given size which uses the
     * given buffer.
     *
     * @param buf Workspace for the work reference. Must be at least
     * `bufBytes(size)` bytes long.
     * @param size Size of the work ensemble to initialize.
     */
    FETA2_DEVICE RefMatrixEnsemble<Scalar_, Rows_, Cols_, ref::WORK> workRef(
        Scalar_* buf, const idx_t& size) const;

    /**
     * @brief Minimum size (in bytes) required for allocation on an arbitrary
     * buffer, if the ensemble were to be constructed for `size` elements.
     */
    FETA2_DEVICE inline static idx_t bufBytes(const idx_t& size);

    /**
     * @brief Returns a pointer to the element following the last element in the
     * scalar array.
     */
    FETA2_DEVICE Scalar_* end() const { return data_ + size_ * Rows_ * Cols_; }

    /** @brief Pointer to the underlying data buffer. */
    FETA2_DEVICE inline Scalar_* data() const { return data_.data(); }

private:
    ScalarRef_ data_;
    idx_t size_;
};

template<typename S, dim_t Rows_, dim_t Cols_, ref::RefKind kind_>
FETA2_DEVICE RefMatrixEnsemble<S, Rows_, Cols_, kind_>::RefMatrixEnsemble(
    ScalarRef_&& data, idx_t size)
    : data_{ std::move(data) }
    , size_{ size }
{
}

template<typename S, dim_t Rows_, dim_t Cols_, ref::RefKind kind_>
FETA2_DEVICE RefMatrixEnsemble<S, Rows_, Cols_, kind_>::RefMatrixEnsemble(
    S* const data, const idx_t size)
    : data_(data, size * Rows_ * Cols_)
    , size_{ size }
{
}

template<typename S, dim_t Rows_, dim_t Cols_, ref::RefKind kind_>
FETA2_DEVICE typename RefMatrixEnsemble<S, Rows_, Cols_, kind_>::ElementMap
RefMatrixEnsemble<S, Rows_, Cols_, kind_>::operator[](const idx_t& idx)
{
    FETA2_ASSERT(idx >= 0 && idx < size_, "index out of bounds!");
    const idx_t innerStride = size_;
    const idx_t outerStride = innerStride * Rows_;
    return ElementMap(
        data_.data() + idx, Rows_, Cols_, StrideT(outerStride, innerStride));
}

template<typename S, dim_t Rows_, dim_t Cols_, ref::RefKind kind_>
FETA2_DEVICE typename RefMatrixEnsemble<S, Rows_, Cols_, kind_>::ElementMap
RefMatrixEnsemble<S, Rows_, Cols_, kind_>::operator[](const SampleIndex& idx)
{
    static_assert(kind_ == ref::GLOBAL || kind_ == ref::WORK,
        "Unsupported reference type!");
    if constexpr (kind_ == ref::GLOBAL) {
        return this->operator[](idx.global());
    } else if constexpr (kind_ == ref::WORK) {
        return this->operator[](idx.work());
    }
}


template<typename S, dim_t Rows_, dim_t Cols_, ref::RefKind kind_>
FETA2_DEVICE inline idx_t RefMatrixEnsemble<S, Rows_, Cols_, kind_>::bufBytes(
    const idx_t& size)
{
    return ScalarRef_::bufBytes(size * Rows_ * Cols_);
}

template<typename S, dim_t Rows_, dim_t Cols_, ref::RefKind kind_>
FETA2_DEVICE RefMatrixEnsemble<S, Rows_, Cols_, ref::WORK>
RefMatrixEnsemble<S, Rows_, Cols_, kind_>::workRef(
    S* buf, const idx_t& size) const
{
    return RefMatrixEnsemble<S, Rows_, Cols_, ref::WORK>(buf, size);
}
} // namespace detail
} // namespace core
} // namespace feta2
