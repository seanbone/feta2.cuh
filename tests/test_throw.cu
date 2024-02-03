#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop
#ifdef __clang__
#pragma pop_macro("__noinline__")
#endif

#include <feta2/feta.cuh>

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
