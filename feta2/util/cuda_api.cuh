#pragma once

// Third-party includes
#include <cuda.h>

// Internal includes
#include "throw.hpp"
#include "typedefs.cuh"

/** @brief Check for errors when making CUDA API calls */
#define FETA2_CUAPI(cuda_retVal)                                               \
    {                                                                          \
        ::feta2::err::detail::checkCudaApiError(                               \
            (cuda_retVal), __FILE__, __LINE__, __func__);                      \
    }

#define FETA2_ASSERT(condition, ...)                                           \
    {                                                                          \
        ::feta2::err::detail::assert_((condition), __FILE__, __LINE__,         \
            __func__, #condition, ##__VA_ARGS__);                              \
    }

#define FETA2_KERNEL_PRE()                                                     \
    {                                                                          \
    }

#define FETA2_KERNEL_POST()                                                    \
    {                                                                          \
    }

#ifdef FETA2_DEBUG_MODE
#define FETA2_GASSERT(condition, ...) assert(condition)
#else
#define FETA2_GASSERT(condition, ...)                                          \
    {                                                                          \
    }
#endif

namespace feta2 {
namespace err {
namespace detail {
static void checkCudaApiError(cudaError_t errorCode, const std::string file,
    int line, const std::string func)
{
    if (errorCode != cudaSuccess) {
        throw_(file, line, func,
            "CUDA API error: ", cudaGetErrorString(errorCode));
    }
}

template<typename... Ts>
FETA2_DEVICE static void assert_(bool condition, const std::string file,
    int line, const std::string func, const std::string conditionStr,
    Ts... extra)
{
#ifdef __CUDA_ARCH__
    assert(condition);
#else
    if (!condition) {
        throw_(file, line, func, "assertion failed: (", conditionStr, ")\n\t",
            extra...);
    }
#endif
}
} // namespace detail
} // namespace err
} // namespace feta2
