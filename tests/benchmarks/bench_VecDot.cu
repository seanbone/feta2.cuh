/** Benchmark of vector dot product operation: out[i] = a[i].dot(b[i]) */

#include "BenchBase.cuh"

namespace feta2_bench {
namespace bench_vecDot {

template<typename dimsHelper_>
class VecDot : public BenchTest {
public:
    static constexpr dim_t Dims = dimsHelper_::dims;

    using Vecs = feta2::VectorEnsemble<Scalar, Dims>;

    VecDot()
        : BenchTest()
        , a(nSamples)
        , b(nSamples)
        , out(nSamples)
    {
        this->a.asyncMemcpyHostToDevice();
        this->b.asyncMemcpyHostToDevice();
        this->out.asyncMemcpyHostToDevice();
        cudaDeviceSynchronize();
    }

    struct VecDotKernelRunner : public KernelRunner {
        VecDotKernelRunner(VecDot<dimsHelper_>& testObj)
            : test_{ testObj }
        {
        }

        virtual idx_t flopsPerSamplePerRep() const override
        {
            return 2 * Dims - 1;
        };

        virtual void run(int reps) const = 0;

        Scalar* a() const { return test_.a.deviceData(); }
        Scalar* b() const { return test_.b.deviceData(); }
        Scalar* out() const { return test_.out.deviceData(); }

        VecDot<dimsHelper_>& test_;
    };

    Vecs a;
    Vecs b;
    Scalars out;
};


// Run each test for multiple vector dimensions
using TestTypes = ::testing::Types<VecDims<3>, VecDims<6>, VecDims<9>>;
TYPED_TEST_SUITE(VecDot, TestTypes);


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

TYPED_TEST(VecDot, FETA2)
{
    using TestT = VecDot<TypeParam>;
    struct VecDot_FETA2 : public TestT::VecDotKernelRunner {
        using TestT::VecDotKernelRunner::VecDotKernelRunner;
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
        out[si] = a[si].dot(b[si]);
}

TYPED_TEST(VecDot, naiveEigen)
{
    using TestT = VecDot<TypeParam>;
    struct VecDot_naiveEigen : public TestT::VecDotKernelRunner {
        using TestT::VecDotKernelRunner::VecDotKernelRunner;
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
        Scalar dot = 0;
        for (int i = 0; i < dims; i++) {
            dot += a[dims * si + i] * b[dims * si + i];
        }
        out[si] = dot;
    }
}

TYPED_TEST(VecDot, manualBadStride)
{
    using TestT = VecDot<TypeParam>;
    struct VecDot_badStride : public TestT::VecDotKernelRunner {
        using TestT::VecDotKernelRunner::VecDotKernelRunner;
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
        Scalar dot = 0;
        for (int i = 0; i < dims; i++) {
            dot += a[i * nSamples + si] * b[i * nSamples + si];
        }
        out[si] = dot;
    }
}

TYPED_TEST(VecDot, manualGoodStride)
{
    using TestT = VecDot<TypeParam>;
    struct VecDot_goodStride : public TestT::VecDotKernelRunner {
        using TestT::VecDotKernelRunner::VecDotKernelRunner;
        void run(int reps) const override
        {
            vecDot_manualGood<Scalar, TestT::Dims>
                <<<TestT::nBlocks, TestT::blockSize>>>(
                    reps, this->a(), this->b(), this->out(), TestT::nSamples);
        }
    };

    this->runBenchmark(VecDot_goodStride(*this));
}

} // namespace bench_vecDot
} // namespace feta2_bench
