#pragma once

// Third-party includes
#include <eigen3/Eigen/Dense>

// Internal includes
#include "util/SampleIndex.cuh"
#include "util/cuda_api.cuh"
#include "util/throw.hpp"
#include "util/typedefs.cuh"

#ifndef FETA2_NOT_FETA
namespace feta = feta2;
#endif
