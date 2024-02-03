#pragma once

#define FETA2_DEVICE __host__ __device__

namespace feta2 {
/** @brief Type used for indexing, eg in arrays */
using idx_t = int;

/** @brief Type used for dimesions, eg of a vector or matrix */
using dim_t = int;

namespace mem {
enum RefType { //
    WORK_REF,
    GLOBAL_REF
};
}
} // namespace feta2
