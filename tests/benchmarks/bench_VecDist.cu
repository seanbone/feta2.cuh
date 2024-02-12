/** Benchmark of vector euclidean distance: out[i] = (a[i] - b[i]).norm() */

#include "BenchBase.cuh"

namespace feta2_bench {
namespace bench_vecDist {

template<typename dimsHelper_>
class BenchVecDist : public BenchTest {
public:
    static constexpr dim_t Dims = dimsHelper_::dims;

    using Vecs = feta2::VectorEnsemble<Scalar, Dims>;

    BenchVecDist()
        : BenchTest()
        , a(nSamples)
        , b(nSamples)
        , out(nSamples)
    {
        std::printf("nDims: %d\n", Dims);
        std::printf("op: vecDist\n");
        this->a.asyncMemcpyHostToDevice();
        this->b.asyncMemcpyHostToDevice();
        this->out.asyncMemcpyHostToDevice();
        cudaDeviceSynchronize();
    }

    struct VecDisttKernelRunner : public KernelRunner {
        VecDisttKernelRunner(BenchVecDist<dimsHelper_>& testObj)
            : test_{ testObj }
        {
        }

        virtual idx_t flopsPerSamplePerRep() const override
        {
            // This is a rough estimate of the relative cost of sqrt
            constexpr idx_t sqrtCost = 5;
            return (3 * Dims - 1) + sqrtCost;
        };

        virtual void run(int reps) const = 0;

        Scalar* a() const { return test_.a.deviceData(); }
        Scalar* b() const { return test_.b.deviceData(); }
        Scalar* out() const { return test_.out.deviceData(); }

        BenchVecDist<dimsHelper_>& test_;
    };

    Vecs a;
    Vecs b;
    Scalars out;
};


// Run each test for multiple vector dimensions
using TestTypes = ::testing::Types<VecDims<3>, VecDims<6>, VecDims<9>>;
TYPED_TEST_SUITE(BenchVecDist, TestTypes);


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
        out[si] = (a[si] - b[si]).norm();
}

TYPED_TEST(BenchVecDist, FETA2)
{
    using TestT = BenchVecDist<TypeParam>;
    struct VecDot_FETA2 : public TestT::VecDisttKernelRunner {
        using TestT::VecDisttKernelRunner::VecDisttKernelRunner;
        void run(int reps) const override
        {
            vecDot_feta2<Scalar, TestT::Dims>
                <<<TestT::nBlocks, TestT::blockSize>>>(reps,
                    { this->a(), TestT::nSamples },
                    { this->b(), TestT::nSamples },
                    { this->out(), TestT::nSamples });
        }
    };

    this->runBenchmark(VecDot_FETA2(*this));
}


template<typename Scalar, dim_t dims>
__global__ void vecDot_naiveEigen(int reps, Eigen::Vector<Scalar, dims>* a,
    Eigen::Vector<Scalar, dims>* b, Scalar* out, int nSamples)
{
    const int si = threadIdx.x + blockIdx.x * blockDim.x;

    if (si >= nSamples)
        return;

    for (int i = 0; i < reps; i++)
        out[si] = (a[si] - b[si]).norm();
}

TYPED_TEST(BenchVecDist, naiveEigen)
{
    using TestT = BenchVecDist<TypeParam>;
    struct VecDot_naiveEigen : public TestT::VecDisttKernelRunner {
        using TestT::VecDisttKernelRunner::VecDisttKernelRunner;
        virtual void run(int reps) const override
        {
            using EigenT = Eigen::Vector<Scalar, TestT::Dims>;
            vecDot_naiveEigen<Scalar, TestT::Dims>
                <<<TestT::nBlocks, TestT::blockSize>>>(reps, (EigenT*)this->a(),
                    (EigenT*)this->b(), this->out(), TestT::nSamples);
        }
    };

    this->runBenchmark(VecDot_naiveEigen(*this));
}


template<typename Scalar, dim_t dims>
__global__ void vecDot_manualBad(
    int reps, Scalar* a, Scalar* b, Scalar* out, int nSamples)
{
    const int si = threadIdx.x + blockIdx.x * blockDim.x;

    if (si >= nSamples)
        return;

    for (int _ = 0; _ < reps; _++) {
        Scalar norm = 0;
        for (int i = 0; i < dims; i++) {
            const Scalar diff = a[dims * si + i] - b[dims * si + i];
            norm += diff * diff;
        }
        out[si] = sqrt(norm);
    }
}

TYPED_TEST(BenchVecDist, manualBadStride)
{
    using TestT = BenchVecDist<TypeParam>;
    struct VecDot_badStride : public TestT::VecDisttKernelRunner {
        using TestT::VecDisttKernelRunner::VecDisttKernelRunner;
        void run(int reps) const override
        {
            vecDot_manualBad<Scalar, TestT::Dims>
                <<<TestT::nBlocks, TestT::blockSize>>>(
                    reps, this->a(), this->b(), this->out(), TestT::nSamples);
        }
    };

    this->runBenchmark(VecDot_badStride(*this));
}


template<typename Scalar, dim_t dims>
__global__ void vecDot_manualGood(
    int reps, Scalar* a, Scalar* b, Scalar* out, int nSamples)
{
    const int si = threadIdx.x + blockIdx.x * blockDim.x;

    if (si >= nSamples)
        return;

    for (int _ = 0; _ < reps; _++) {
        if constexpr (dims == 3) {
            int i          = si;
            const Scalar x = a[i] - b[i];
            i += nSamples;
            const Scalar y = a[i] - b[i];
            i += nSamples;
            const Scalar z = a[i] - b[i];
            out[si]        = sqrt(x * x + y * y + z * z);
            // Oddly enough this seems significantly slower
            // out[si]  = norm3d(x, y, z);
        } else {
            Scalar norm = 0;
            for (int i = 0; i < dims; i++) {
                const Scalar diff = a[i * nSamples + si] - b[i * nSamples + si];
                norm += diff * diff;
            }
            out[si] = sqrt(norm);
        }
    }
}

TYPED_TEST(BenchVecDist, manualGoodStride)
{
    using TestT = BenchVecDist<TypeParam>;
    struct VecDot_goodStride : public TestT::VecDisttKernelRunner {
        using TestT::VecDisttKernelRunner::VecDisttKernelRunner;
        void run(int reps) const override
        {
            vecDot_manualGood<Scalar, TestT::Dims>
                <<<TestT::nBlocks, TestT::blockSize>>>(
                    reps, this->a(), this->b(), this->out(), TestT::nSamples);
        }
    };

    this->runBenchmark(VecDot_goodStride(*this));
}

} // namespace bench_vecDist
} // namespace feta2_bench
