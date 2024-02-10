#pragma once

#include <random>

#include <gtest/gtest.h>

#include <feta2/feta.cuh>

namespace feta2_tests {
/** @brief Base class for test suites */
class Test : public ::testing::Test {
protected:
    using idx_t = feta2::idx_t;

    static constexpr idx_t testSize_  = 101;
    static constexpr idx_t blockSize_ = 32;
    static constexpr idx_t nBlocks_ = (testSize_ + blockSize_ - 1) / blockSize_;

public:
    Test()
        : rng_(rd_())
        , uni_(0., 1.)
    {
    }


protected:
    /** @brief Random `double` sampled from a uniform distribution [-1; 1) */
    double rand_() { return 2 * uni_(rng_) - 1; }

    // Random seed
    std::random_device rd_;
    // Random Number Generator
    std::mt19937 rng_;
    // Uniform distribution
    std::uniform_real_distribution<double> uni_;
};
} // namespace feta2_tests
