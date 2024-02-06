/**
 * Test suite to verify the functionality of MatrixEnsemble, which handles
 * memory management on host and device for arrays of Eigen matrices.
 */

#include "TestBase.cuh"
#include "feta2/feta.cuh"
#include "feta2/util/err.cuh"
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

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
        : matrices_(testSize)
        , ensemble_(testSize)
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


} // namespace MatrixEnsembleTest
} // namespace feta2_tests
