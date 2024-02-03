#pragma once

// STL includes
#include <sstream>

// Third-party includes
#include <cuda.h>

#include "typedefs.cuh"

/** @brief Macro used to throw exceptions. For host code only. */
#define FETA2_HTHROW(...)                                                      \
    ::feta2::err::detail::throw_(__FILE__, __LINE__, __func__, ##__VA_ARGS__)

/** @brief Check for errors when making CUDA API calls. Host only! */
#define FETA2_CUAPI(cuda_retVal)                                               \
    {                                                                          \
        ::feta2::err::detail::checkCudaApiError(                               \
            (cuda_retVal), __FILE__, __LINE__, __func__);                      \
    }

/** @brief Check condition. Callable from host or device code. */
#define FETA2_ASSERT(condition, ...)                                           \
    {                                                                          \
        ::feta2::err::detail::assert_((condition), __FILE__, __LINE__,         \
            __func__, #condition, ##__VA_ARGS__);                              \
    }

#ifdef FETA2_DEBUG_MODE
/** @brief Check condition if in debug mode. Callable from host or device. */
#define FETA2_DBG_ASSERT(condition, ...)                                       \
    {                                                                          \
        ::feta2::err::detail::assert_((condition), __FILE__, __LINE__,         \
            __func__, #condition, ##__VA_ARGS__);                              \
    }

#define FETA2_KERNEL_PRE()                                                     \
    {                                                                          \
    }

#define FETA2_KERNEL_POST()                                                    \
    {                                                                          \
        ::feta2::err::detail::checkCudaApiError(                               \
            cudaDeviceSynchronize(), __FILE__, __LINE__, __func__);            \
        ::feta2::err::detail::checkCudaApiError(                               \
            cudaGetLastError(), __FILE__, __LINE__, __func__);                 \
    }

#else // FETA2_DEBUG_MODE

#define FETA2_DBG_ASSERT(condition, ...)                                       \
    {                                                                          \
    }

#define FETA2_KERNEL_PRE()                                                     \
    {                                                                          \
    }

#define FETA2_KERNEL_POST()                                                    \
    {                                                                          \
    }
#endif // FETA2_DEBUG_MODE


namespace feta2 {
namespace err {
namespace detail {

// Concatenate variable number of arguments into a string
template<typename... Ts>
std::string cat(Ts&&... args)
{
    // Uses C++17 fold expression
    std::ostringstream oss;
    (oss << ... << std::forward<Ts>(args));
    return oss.str();
}

template<typename... Ts>
void throw_(
    const std::string file, int line, const std::string func, Ts... params)
{
    throw std::runtime_error(
        cat("[", file, ":", line, ":", func, "] FETA error: ", params...));
}


static void checkCudaApiError(cudaError_t errorCode, const std::string file,
    int line, const std::string func)
{
    if (errorCode != cudaSuccess) {
        throw_(file, line, func,
            "CUDA API error: ", cudaGetErrorString(errorCode));
    }
}

template<typename... Ts>
FETA2_DEVICE static void assert_(bool condition, const char* file, int line,
    const char* func, const char* conditionStr, Ts... extra)
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
