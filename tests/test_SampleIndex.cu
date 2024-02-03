#include "TestBase.cuh"

namespace feta2_tests {
using feta2::SampleIndex;

TEST(SampleIndexTest, SimpleTest)
{
    const SampleIndex si(1, 1, 10);
    EXPECT_EQ(si.global(), 11);
    EXPECT_EQ(si.work(), 1);
}

TEST(SampleIndexTest, Clone)
{
    const SampleIndex si(1, 1, 10);
    const SampleIndex si2 = si.clone();
    EXPECT_EQ(si2.global(), 11);
    EXPECT_EQ(si2.work(), 1);
}

TEST(SampleIndexTest, FromInt)
{
    const SampleIndex si(99);
    EXPECT_EQ(si.global(), 99);
    EXPECT_EQ(si.work(), 99);
}
} // namespace feta2_tests
