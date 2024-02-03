#include "TestBase.cuh"

namespace feta2_tests {
namespace TestThrow {
TEST(TestThrow, TestThrow)
{
    EXPECT_ANY_THROW(FETA2_HTHROW("HELLO THERE ", 42));
}

TEST(TestThrow, TestCudaApi)
{
    EXPECT_ANY_THROW(FETA2_CUAPI(cudaMallocHost(nullptr, 0)));
}

TEST(TestThrow, TestAssertion)
{
    EXPECT_ANY_THROW(FETA2_ASSERT(false, "what gives man?"));
}
} // namespace TestThrow
} // namespace feta2_tests
