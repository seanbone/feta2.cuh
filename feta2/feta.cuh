#pragma once

// Internal includes
#include "core/MatrixEnsemble.cuh"
#include "core/ScalarEnsemble.cuh"
#include "util.cuh"

namespace feta2 {
using core::MatrixEnsemble;
using core::ScalarEnsemble;
} // namespace feta2

#ifndef FETA2_NOT_FETA
namespace feta = feta2;
#endif
