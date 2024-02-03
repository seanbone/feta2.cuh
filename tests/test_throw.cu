#include "TestBase.cuh"

namespace feta2_tests {
namespace TestThrow {
TEST(TestThrow, TestThrow)
{
    EXPECT_THROW(FETA2_HTHROW("HELLO THERE ", 42), std::runtime_error);
}

TEST(TestThrow, TestCudaApi)
{
    EXPECT_THROW(FETA2_CUAPI(cudaMallocHost(nullptr, 0)), std::runtime_error);
}
} // namespace TestThrow
} // namespace feta2_tests
