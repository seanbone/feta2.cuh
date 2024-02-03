#pragma once

// Third-party includes
#include <cuda.h>

// Internal includes
#include "throw.hpp"

/** @brief Check for errors when making CUDA API calls */
#define FETA2_CUAPI(cuda_retVal)                                               \
    {                                                                          \
        ::feta2::err::detail::checkCudaApiError(                               \
            (cuda_retVal), __FILE__, __LINE__, __func__);                      \
    }

namespace feta2 {
namespace err {
namespace detail {
__host__ static void checkCudaApiError(cudaError_t errorCode,
    const std::string file, int line, const std::string func)
{
    if (errorCode != cudaSuccess) {
        throw_(file, line, func,
            "CUDA API error: ", cudaGetErrorString(errorCode));
    }
}
} // namespace detail
} // namespace err
} // namespace feta2
