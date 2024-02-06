#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <eigen3/Eigen/Dense>

#define FETA2_DEVICE __host__ __device__

namespace feta2 {
/** @brief Type used for indexing, eg in arrays */
using idx_t = Eigen::Index;

/** @brief Type used for dimesions, eg of a vector or matrix */
using dim_t = Eigen::Index;

namespace core {
namespace ref {
enum RefKind { //
    WORK,
    GLOBAL
};
}
} // namespace core
} // namespace feta2
