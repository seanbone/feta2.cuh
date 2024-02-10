
#include "TestBase.cuh"
#include <feta2/feta.cuh>

namespace feta2_tests {
namespace bench_vecDot {

using feta2::dim_t;

template<feta2::dim_t Dims_>
struct VecDims {
    static constexpr dim_t dims = Dims_;
};

template<typename dimsHelper_>
class BenchVecDot : public Test {
public:
    static constexpr dim_t Dims = dimsHelper_::dims;

    using Scalar  = double;
    using Vecs    = feta2::VectorEnsemble<Scalar, Dims>;
    using Scalars = feta2::ScalarEnsemble<Scalar>;

    static constexpr idx_t testSize  = 1e6;
    static constexpr idx_t nReps     = 1e4;
    static constexpr idx_t blockSize = 32;
    static constexpr idx_t nBlocks   = (testSize + blockSize - 1) / blockSize;

    BenchVecDot()
        : a(testSize)
        , b(testSize)
        , out(testSize)
    {
    }

    class KernelRunner {
    public:
        virtual int flopsPerSamplePerRep() const                            = 0;
        virtual void run(int reps, Scalar* a, Scalar* b, Scalar* out) const = 0;
    };

    class VecDotKernelRunner : public KernelRunner {
    public:
        virtual idx_t flopsPerSamplePerRep() const override
        {
            return 2 * Dims - 1;
        };
        virtual void run(int reps, Scalar* a, Scalar* b, Scalar* out) const = 0;
    };

    void runBenchmark(const KernelRunner& runner)
    {
        this->a.asyncMemcpyHostToDevice();
        this->b.asyncMemcpyHostToDevice();
        this->out.asyncMemcpyHostToDevice();
        cudaDeviceSynchronize();

        std::cout << "Warmup..." << std::endl;
        runner.run(
            nReps / 10, a.deviceData(), b.deviceData(), out.deviceData());

        std::cout << "Benchmarking..." << std::endl;
        cudaEvent_t start, stop;
        float time;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        runner.run(nReps, a.deviceData(), b.deviceData(), out.deviceData());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);


        const double totalFlops
            = (double)runner.flopsPerSamplePerRep() * nReps * testSize;
        std::cout << "MFLOP/s: " << totalFlops / time / 1e6 << std::endl;
    }

    Vecs a;
    Vecs b;
    Scalars out;
};

using TestTypes = ::testing::Types<VecDims<3>, VecDims<6>, VecDims<9>>;
TYPED_TEST_SUITE(BenchVecDot, TestTypes);


template<typename Scalar, dim_t dims>
__global__ void vecDot_feta2(int reps,
    typename feta2::VectorEnsemble<Scalar, dims>::GRef a,
    typename feta2::VectorEnsemble<Scalar, dims>::GRef b,
    typename feta2::ScalarEnsemble<Scalar>::GRef out)
{
    feta2::SampleIndex si(threadIdx.x, blockIdx.x, blockDim.x);

    if (si.global() >= out.size())
        return;

    for (int i = 0; i < reps; i++)
        out[si] = a[si].dot(b[si]);
}

TYPED_TEST(BenchVecDot, FETA2)
{
    using TestT = BenchVecDot<TypeParam>;

    class VecDot_FETA2 : public TestT::VecDotKernelRunner {
    public:
        using Scalar = typename TestT::Scalar;

        virtual void run(
            int reps, Scalar* a, Scalar* b, Scalar* out) const override
        {
            vecDot_feta2<Scalar, TestT::Dims>
                <<<TestT::nBlocks, TestT::blockSize>>>(reps,
                    { a, TestT::testSize }, { b, TestT::testSize },
                    { out, TestT::testSize });
        }
    };

    this->runBenchmark(VecDot_FETA2());
}


template<typename Scalar, dim_t dims>
__global__ void vecDot_naiveEigen(int reps, Eigen::Vector<Scalar, dims>* a,
    Eigen::Vector<Scalar, dims>* b, Scalar* out, int nSamples)
{
    const int si = threadIdx.x + blockIdx.x * blockDim.x;

    if (si >= nSamples)
        return;

    for (int i = 0; i < reps; i++)
        out[si] = a[si].dot(b[si]);
}

TYPED_TEST(BenchVecDot, naiveEigen)
{
    using TestT = BenchVecDot<TypeParam>;

    class VecDot_naiveEigen : public TestT::VecDotKernelRunner {
    public:
        using Scalar = typename TestT::Scalar;

        virtual void run(
            int reps, Scalar* a, Scalar* b, Scalar* out) const override
        {
            using EigenT = Eigen::Vector<Scalar, TestT::Dims>;
            vecDot_naiveEigen<Scalar, TestT::Dims>
                <<<TestT::nBlocks, TestT::blockSize>>>(
                    reps, (EigenT*)a, (EigenT*)b, out, TestT::testSize);
        }
    };

    this->runBenchmark(VecDot_naiveEigen());
}


template<typename Scalar, dim_t dims>
__global__ void vecDot_manualBad(
    int reps, Scalar* a, Scalar* b, Scalar* out, int nSamples)
{
    const int si = threadIdx.x + blockIdx.x * blockDim.x;

    if (si >= nSamples)
        return;

    for (int _ = 0; _ < reps; _++) {
        Scalar dot = 0;
        for (int i = 0; i < dims; i++) {
            dot += a[dims * si + i] * b[dims * si + i];
        }
        out[si] = dot;
    }
}

TYPED_TEST(BenchVecDot, manualBadStride)
{
    using TestT = BenchVecDot<TypeParam>;

    class VecDot_badStride : public TestT::VecDotKernelRunner {
    public:
        using Scalar = typename TestT::Scalar;

        virtual void run(
            int reps, Scalar* a, Scalar* b, Scalar* out) const override
        {
            vecDot_manualBad<Scalar, TestT::Dims>
                <<<TestT::nBlocks, TestT::blockSize>>>(
                    reps, a, b, out, TestT::testSize);
        }
    };

    this->runBenchmark(VecDot_badStride());
}


template<typename Scalar, dim_t dims>
__global__ void vecDot_manualGood(
    int reps, Scalar* a, Scalar* b, Scalar* out, int nSamples)
{
    const int si = threadIdx.x + blockIdx.x * blockDim.x;

    if (si >= nSamples)
        return;

    for (int _ = 0; _ < reps; _++) {
        Scalar dot = 0;
        for (int i = 0; i < dims; i++) {
            dot += a[i * nSamples + si] * b[i * nSamples + si];
        }
        out[si] = dot;
    }
}

TYPED_TEST(BenchVecDot, manualGoodStride)
{
    using TestT = BenchVecDot<TypeParam>;

    class VecDot_goodStride : public TestT::VecDotKernelRunner {
    public:
        using Scalar = typename TestT::Scalar;

        virtual void run(
            int reps, Scalar* a, Scalar* b, Scalar* out) const override
        {
            vecDot_manualGood<Scalar, TestT::Dims>
                <<<TestT::nBlocks, TestT::blockSize>>>(
                    reps, a, b, out, TestT::testSize);
        }
    };

    this->runBenchmark(VecDot_goodStride());
}

} // namespace bench_vecDot
} // namespace feta2_tests
