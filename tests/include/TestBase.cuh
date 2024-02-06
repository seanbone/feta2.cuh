#pragma once

#include <random>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop
#ifdef __clang__
#pragma pop_macro("__noinline__")
#endif

#include <feta2/feta.cuh>

namespace feta2_tests {
/** @brief Base class for test suites */
class Test : public ::testing::Test {
protected:
    using idx_t = feta2::idx_t;

    static constexpr idx_t testSize  = 101;
    static constexpr idx_t blockSize = 32;
    static constexpr idx_t nBlocks   = (testSize + blockSize - 1) / blockSize;

public:
    Test()
        : rng_(rd_())
        , uni_(0., 1.)
    {
    }


protected:
    /** @brief Random `double` sampled from a uniform distribution [0; 1) */
    double rand_() { return uni_(rng_); }

    // Random seed
    std::random_device rd_;
    // Random Number Generator
    std::mt19937 rng_;
    // Uniform distribution
    std::uniform_real_distribution<double> uni_;
};
} // namespace feta2_tests
