/**
 * Test suite to verify the functionality of MatrixEnsemble, which handles
 * memory management on host and device for arrays of Eigen matrices.
 */

#include "TestBase.cuh"

namespace feta2_tests {
namespace MatrixEnsembleTest {

using Scalar         = double;
using ScalarEnsemble = feta2::ScalarEnsemble<Scalar>;
using Ensemble       = feta2::MatrixEnsemble<Scalar, 3, 3>;

auto normOp = [](const Ensemble::Element& element) { return element.norm(); };
auto doubleNormOp
    = [](const Ensemble::Element& element) { return 2 * element.norm(); };

class MatrixEnsembleTest : public Test {
protected:
    std::vector<Ensemble::Element> matrices_;

    Ensemble ensemble_;

public:
    MatrixEnsembleTest()
        : matrices_(testSize_)
        , ensemble_(testSize_)
    {
        for (idx_t i = 0; i < ensemble_.size(); i++) {
            matrices_[i] = rand_() * Ensemble::Element::Identity();
            ensemble_[i] = matrices_[i];
        }
    }

    template<typename F>
    void validate(const F& op, const feta2::ScalarEnsemble<Scalar>& results)
    {
        for (idx_t i = 0; i < ensemble_.size(); i++) {
            ASSERT_DOUBLE_EQ(op(matrices_[i]), results[i]) << "i = " << i;
        }
    }

    ScalarEnsemble hostNorms(const Ensemble& e)
    {
        ScalarEnsemble norms(e.size());
        for (idx_t i = 0; i < e.size(); i++) {
            norms[i] = e[i].norm();
        }
        return norms;
    }
};

TEST_F(MatrixEnsembleTest, SimpleNorm)
{
    validate(normOp, hostNorms(ensemble_));
}

TEST_F(MatrixEnsembleTest, MoveSemantics)
{
    Ensemble b(std::move(ensemble_));

    Ensemble c = std::move(b);

    ASSERT_EQ(c.size(), matrices_.size());
    validate(normOp, hostNorms(c));
}


TEST_F(MatrixEnsembleTest, DeviceCopy)
{
    ensemble_.asyncMemcpyHostToDevice();
    FETA2_CUAPI(cudaDeviceSynchronize());

    // Double each matrix on host array
    for (idx_t i = 0; i < ensemble_.size(); i++) {
        ensemble_[i] *= 2;
    }
    validate(doubleNormOp, hostNorms(ensemble_));

    // Overwrite host array with originals
    Ensemble b(std::move(ensemble_));
    b.asyncMemcpyDeviceToHost();
    FETA2_CUAPI(cudaDeviceSynchronize());

    validate(normOp, hostNorms(b));
}


__global__ void doubleEnsemble(Ensemble::GRef ensemble)
{
    feta2::SampleIndex si(threadIdx.x, blockIdx.x, blockDim.x);

    if (si.global() >= ensemble.size())
        return;

    ensemble[si] *= 2;
}

TEST_F(MatrixEnsembleTest, OnDevice_DoubleEnsemble)
{
    EXPECT_ANY_THROW(ensemble_.deviceRef());

    ensemble_.asyncMemcpyHostToDevice();
    FETA2_CUAPI(cudaDeviceSynchronize());

    // Double each matrix on device array
    FETA2_KERNEL_PRE();
    doubleEnsemble<<<blockSize_, nBlocks_>>>(ensemble_.deviceRef());
    FETA2_KERNEL_POST();

    Ensemble b(std::move(ensemble_));
    b.asyncMemcpyDeviceToHost();
    FETA2_CUAPI(cudaDeviceSynchronize());

    validate(doubleNormOp, hostNorms(b));
}


#ifdef FETA2_DEBUG_MODE
TEST_F(MatrixEnsembleTest, OnHost_OutOfBounds_Throws)
{
    EXPECT_ANY_THROW(ensemble_[ensemble_.size()]);
    EXPECT_ANY_THROW(ensemble_.hostRef()[feta2::SampleIndex(ensemble_.size())]);
}
#endif


__global__ void doubleEnsembleShMem(Ensemble::GRef ensemble)
{
    feta2::SampleIndex si(threadIdx.x, blockIdx.x, blockDim.x);

    if (si.global() >= ensemble.size())
        return;

    extern __shared__ Scalar smem[];
    Ensemble::WRef workEnsemble = ensemble.workRef(smem, blockDim.x);

    workEnsemble[si] = ensemble[si];
    workEnsemble[si] += ensemble[si];
    ensemble[si] = workEnsemble[si];
}

TEST_F(MatrixEnsembleTest, OnDevice_DoubleEnsembleInShared)
{
    EXPECT_ANY_THROW(ensemble_.deviceRef());

    ensemble_.asyncMemcpyHostToDevice();
    FETA2_CUAPI(cudaDeviceSynchronize());

    // Double each matrix on device array
    FETA2_KERNEL_PRE();
    const idx_t shMemSize = Ensemble::WRef::bufBytes(blockSize_);
    doubleEnsembleShMem<<<blockSize_, nBlocks_, shMemSize>>>(
        ensemble_.deviceRef());
    FETA2_KERNEL_POST();

    Ensemble b(std::move(ensemble_));
    b.asyncMemcpyDeviceToHost();
    FETA2_CUAPI(cudaDeviceSynchronize());

    validate(doubleNormOp, hostNorms(b));
}


} // namespace MatrixEnsembleTest
} // namespace feta2_tests
