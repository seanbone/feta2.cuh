#pragma once

#include "typedefs.cuh"

namespace feta2 {
/**
 * @brief Sample index. Used to dynamically distinguish between work and global
 * indexing (usually this corresponds to global and shared GPU memory).
 */
class SampleIndex {
public:
    FETA2_DEVICE SampleIndex(
        const idx_t& threadIdxx, const idx_t& blockIdxx, const idx_t& blockDimx)
        : global_{ threadIdxx + blockIdxx * blockDimx }
        , work_{ threadIdxx }
    {
    }

    /** @brief Implicit conversion from `idx_t`.
     *
     * In this case work and global indices are the same. Equivalent to
     * `SampleIndex(i, 0, 0)`.
     */
    FETA2_DEVICE SampleIndex(const idx_t& index)
        : global_{ index }
        , work_{ index }
    {
    }

    /** @brief Return the global sample index */
    FETA2_DEVICE inline idx_t global() const { return global_; }

    /** @brief Return the sample index for work arrays. */
    FETA2_DEVICE inline idx_t work() const { return work_; }

    /** @brief Explicit alternative to the copy constructor, which is private.*/
    FETA2_DEVICE SampleIndex clone() const { return SampleIndex(*this); }

private:
    /**
     * Prevent implicit copies by making the copy constructor private.
     * It is still available explicitly through .clone()
     */
    __host__ __device__ SampleIndex(const SampleIndex& other)
        : global_{ other.global_ }
        , work_{ other.work_ }
    {
    }

    const idx_t global_;
    const idx_t work_;
};
} // namespace feta2
