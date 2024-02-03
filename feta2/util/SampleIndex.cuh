#pragma once

#include "typedefs.cuh"

namespace feta2 {
/**
 * @brief Linear sample index.
 *
 * Used to dynamically distinguish between work and global
 * indexing (usually this corresponds to global and shared GPU memory).
 *
 * Note: this assumes linear indexing (as opposed to 2D or 3D)!
 */
class SampleIndex {
public:
    /** @brief Constructor from thread indices.
     *
     * In kernel code, this is usually initialized from the GPU thread indices,
     * e.g.
     * ```
     * const feta::SampleIndex si(threadIdx.x, blockIdx.x, blockDim.x)
     * ```
     *
     * Note: `SampleIndex` assumes linear indexing (as opposed to 2D or 3D)!
     *
     * @param threadIdxx Thread index within its block (`threadIdx.x`)
     * @param blockIdxx Block index within the grid (`blockIdx.x`)
     * @param blockDimx Dimension of the block (`blockDim.x`)
     */
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
    FETA2_DEVICE SampleIndex(const SampleIndex& other)
        : global_{ other.global_ }
        , work_{ other.work_ }
    {
    }

    const idx_t global_;
    const idx_t work_;
};
} // namespace feta2
