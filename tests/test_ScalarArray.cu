/**
 * Test suite to verify the functionality of ScalarArray, which handles memory
 * management on host and device (and transfers between them).
 */
#include "TestBase.cuh"

namespace feta2_tests {
namespace ScalarArrayTest {

using feta2::idx_t;
using feta2::SampleIndex;
using DoubleArray = feta2::ScalarArray<double>;

class ScalarArrayTest : public Test {
protected:
    static constexpr idx_t testSize  = 101;
    static constexpr idx_t blockSize = 32;
    static constexpr idx_t nBlocks   = (testSize + blockSize - 1) / blockSize;

    std::vector<double> data;
    DoubleArray array;

public:
    ScalarArrayTest()
        : data(testSize)
        , array(testSize)
    {
        for (uint i = 0; i < testSize; i++) {
            data[i]  = rand_();
            array[i] = data[i];
        }
    }

    template<typename Iterator, typename FUNC>
    void validateData(Iterator it, FUNC& f)
    {
        for (uint i = 0; i < testSize; i++) {
            ASSERT_DOUBLE_EQ(it[i], f(data[i])) << "i = " << i;
        }
    }
};

TEST_F(ScalarArrayTest, OnHost_MoveCopySemantics)
{
    DoubleArray b(std::move(array));

    auto noop = [](double x) { return x; };
    validateData(b.getHostData(), noop);

    DoubleArray c = std::move(b);
    validateData(c.getHostData(), noop);
}

TEST_F(ScalarArrayTest, CopyToDeviceAndBack_IsCorrect)
{
    // DeviceToHost before HostToDevice is an error
    EXPECT_THROW(array.asyncMemcpyDeviceToHost(), std::runtime_error);
    array.asyncMemcpyHostToDevice();
    DoubleArray b = std::move(array);

    // Double elements on host array
    for (idx_t i = 0; i < b.size(); i++)
        b[i] *= 2;

    // Rewrite original values from device array
    b.asyncMemcpyDeviceToHost();
    FETA2_CUAPI(cudaDeviceSynchronize());
    auto noop = [](double x) { return x; };
    validateData(b.getHostData(), noop);
}


__global__ void doubleArrayCUDA(DoubleArray::GRef array)
{
    const SampleIndex i(threadIdx.x, blockIdx.x, blockDim.x);
    DoubleArray::GRef arr = array;

    if (i.global() >= arr.size())
        return;

    arr[i] *= 2;
}

TEST_F(ScalarArrayTest, EditOnDevice_IsCorrect)
{
    EXPECT_THROW(array.asyncMemcpyDeviceToHost(), std::runtime_error);
    array.asyncMemcpyHostToDevice();
    DoubleArray b = std::move(array);

    FETA2_KERNEL_PRE();
    doubleArrayCUDA<<<nBlocks, blockSize>>>(b.deviceRef());
    FETA2_KERNEL_POST();

    b.asyncMemcpyDeviceToHost();
    FETA2_CUAPI(cudaDeviceSynchronize());
    auto mul2 = [](double x) { return 2 * x; };
    validateData(b.getHostData(), mul2);
}

__global__ void doubleArrayCUDAShared(DoubleArray::GRef arr)
{
    const SampleIndex i(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= arr.size())
        return;

    extern __shared__ double shMem[];
    DoubleArray::WRef shArr = arr.workRef(shMem, blockDim.x);

    shArr[i] = arr[i];
    shArr[i] *= 2;
    arr[i] = shArr[i];
}

TEST_F(ScalarArrayTest, EditOnDeviceShared_IsCorrect)
{
    EXPECT_THROW(array.asyncMemcpyDeviceToHost(), std::runtime_error);
    array.asyncMemcpyHostToDevice();
    DoubleArray b = std::move(array);

    FETA2_KERNEL_PRE();
    idx_t shMem = blockSize * sizeof(double);
    doubleArrayCUDAShared<<<nBlocks, blockSize, shMem>>>(b.deviceRef());
    FETA2_KERNEL_POST();

    b.asyncMemcpyDeviceToHost();
    cudaDeviceSynchronize();
    auto mul2 = [](double x) { return 2 * x; };
    validateData(b.getHostData(), mul2);
}

TEST_F(ScalarArrayTest, OnInit_AllValuesInitialized)
{
    double initVal = 3.14;
    DoubleArray blankArray(testSize, 3.14);

    for (uint i = 0; i < testSize; i++) {
        ASSERT_EQ(blankArray[i], initVal);
    }
}

#ifdef FETA2_DEBUG_MODE
TEST_F(ScalarArrayTest, OnHost_OutOfBoundsAccess_Throws)
{
    EXPECT_ANY_THROW(array[array.size()]);
    EXPECT_ANY_THROW(array[-1]);
    EXPECT_ANY_THROW(array.hostRef()[array.size()]);
    EXPECT_ANY_THROW(array.hostRef()[-1]);
}

TEST_F(ScalarArrayTest, OnNoMemCopy_DeviceRef_Throws)
{
    EXPECT_THROW(array.deviceRef(), std::runtime_error);
}
#endif
} // namespace ScalarArrayTest
} // namespace feta2_tests
