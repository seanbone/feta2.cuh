#pragma once

#include "../util.cuh"

namespace feta2 {
namespace core {
namespace detail {
/**
 * @brief Non-owning reference to an ensemble of scalar values.
 *
 * `RefScalarEnsemble` does not allocate or deallocate any memory. It is
 * constructed from a pointer and it assumes that sufficient memory is allocated
 * there.
 *
 * @tparam S Data type of the entried of the scalar ensemble.
 * @tparam kind Reference kind. If WORK, element access with a `SampleIndex`
 *  will use `SampleIndex::work()`. Otherwise, it uses `SampleIndex::global()`.
 **/
template<typename Scalar_, ref::RefKind kind>
class RefScalarEnsemble {
public:
    /**
     * @brief Construct a non-owning scalar ensemble which assumes `data` has
     * been allocated for at least `size` elements.
     */
    FETA2_DEVICE RefScalarEnsemble(Scalar_* const data, const idx_t size);

    /**
     * @brief Element access operator. Performs bounds-check if
     * `FETA_DEBUG_MODE` is set.
     */
    FETA2_DEVICE Scalar_& operator[](const idx_t& idx) const;

    /**
     * @brief Element access operator. Switches between `SampleIndex::global`
     * and `SampleIndex::work` based on the value of template parameter `work`.
     */
    FETA2_DEVICE Scalar_& operator[](const SampleIndex& idx) const;

    /** @brief Returns the number of elements in this ensemble. */
    FETA2_DEVICE inline idx_t size() const { return size_; }

    /**
     * @brief Returns a pointer to the element following the last element in the
     * scalar ensemble.
     */
    FETA2_DEVICE Scalar_* end() const { return data_ + size_; }

    /**
     * @brief Minimum size (in bytes) required for allocation on an arbitrary
     * buffer, if the ensemble were to be constructed for `size` elements.
     */
    FETA2_DEVICE inline static idx_t bufBytes(const idx_t& size);

    /**
     * @brief Returns a work ensemble reference of the given size which uses the
     * given buffer.
     *
     * @param buf Workspace for the work reference. Must be at least
     * `bufBytes(size)` bytes long.
     * @param size Size of the work ensemble to initialize.
     */
    FETA2_DEVICE RefScalarEnsemble<Scalar_, ref::WORK> workRef(
        Scalar_* buf, const idx_t& size) const;

    /** @brief Pointer to the underlying data buffer. */
    FETA2_DEVICE inline Scalar_* data() const { return data_; }

private:
    Scalar_* data_;
    idx_t size_;
};


/** METHOD IMPLEMENTATIONS **/


template<typename S, ref::RefKind kind>
FETA2_DEVICE S& RefScalarEnsemble<S, kind>::operator[](const idx_t& idx) const
{
    FETA2_DBG_ASSERT(
        idx >= 0 && idx < size_, "RefScalarEnsemble: out of bounds access");
    return data_[idx];
}

template<typename S, ref::RefKind kind>
FETA2_DEVICE S& RefScalarEnsemble<S, kind>::operator[](
    const SampleIndex& idx) const
{
    if constexpr (kind == ref::WORK) {
        return (*this)[idx.work()];
    } else if constexpr (kind == ref::GLOBAL) {
        return (*this)[idx.global()];
    }
}

template<typename S, ref::RefKind kind>
FETA2_DEVICE RefScalarEnsemble<S, kind>::RefScalarEnsemble(
    S* const data, const idx_t size)
    : data_{ data }
    , size_{ size }
{
}


template<typename S, ref::RefKind kind>
FETA2_DEVICE inline idx_t RefScalarEnsemble<S, kind>::bufBytes(
    const idx_t& size)
{
    return size * sizeof(S);
}

template<typename S, ref::RefKind kind>
FETA2_DEVICE RefScalarEnsemble<S, ref::WORK>
RefScalarEnsemble<S, kind>::workRef(S* buf, const idx_t& size) const
{
    return RefScalarEnsemble<S, ref::WORK>(buf, size);
}
} // namespace detail
} // namespace core
} // namespace feta2
