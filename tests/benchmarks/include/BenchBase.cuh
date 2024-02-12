#pragma once

#include <random>

#include <gtest/gtest.h>

#include <feta2/feta.cuh>

namespace feta2_bench {
using feta2::dim_t;
using Scalar = double;

template<feta2::dim_t Dims_>
struct VecDims {
    static constexpr dim_t dims = Dims_;
};

/** @brief Base class for test suites */
class BenchTest : public ::testing::Test {
protected:
    using idx_t   = feta2::idx_t;
    using dim_t   = feta2::dim_t;
    using Scalars = feta2::ScalarEnsemble<Scalar>;

    static constexpr idx_t nSamples  = 1e4;
    static constexpr idx_t nReps     = 1e2;
    static constexpr idx_t blockSize = 32;
    static constexpr idx_t nBlocks   = (nSamples + blockSize - 1) / blockSize;

public:
    BenchTest()
    {
        std::printf("nSamples: %.2e\n", (float)nSamples);
        std::printf("nReps: %.2e\n", (float)nReps);
    }

    struct KernelRunner {
        virtual int flopsPerSamplePerRep() const = 0;
        virtual void run(int reps) const         = 0;
    };

    void runBenchmark(const KernelRunner& runner)
    {
        std::cout << "Warmup..." << std::endl;
        runner.run(nReps / 10);

        std::cout << "Benchmarking..." << std::endl;
        cudaEvent_t start, stop;
        float timeMilli;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        runner.run(nReps);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&timeMilli, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::cout << "* Avg ms/rep: " << timeMilli / nReps << std::endl;
        const float timeSec = timeMilli / 1e3;
        const double totalFlops
            = (double)runner.flopsPerSamplePerRep() * nReps * nSamples;
        std::cout << "* GFLOP/s: " << totalFlops / timeSec / 1e9 << std::endl;
    }
};
} // namespace feta2_bench
