#pragma once

// Internal includes
#include "core/MatrixEnsemble.cuh"
#include "core/ScalarEnsemble.cuh"
#include "util.cuh"

namespace feta2 {
using core::ScalarEnsemble;
using DoubleEnsemble = ScalarEnsemble<double>;
using FloatEnsemble  = ScalarEnsemble<float>;

using core::MatrixEnsemble;

template<typename Scalar>
using Matrix2Ensemble  = MatrixEnsemble<Scalar, 2, 2>;
using Matrix2dEnsemble = Matrix2Ensemble<double>;
using Matrix2fEnsemble = Matrix2Ensemble<float>;
using Matrix2iEnsemble = Matrix2Ensemble<int>;

template<typename Scalar>
using Matrix3Ensemble  = MatrixEnsemble<Scalar, 3, 3>;
using Matrix3dEnsemble = Matrix3Ensemble<double>;
using Matrix3fEnsemble = Matrix3Ensemble<float>;
using Matrix3iEnsemble = Matrix3Ensemble<int>;

template<typename Scalar>
using Matrix4Ensemble  = MatrixEnsemble<Scalar, 4, 4>;
using Matrix4dEnsemble = Matrix4Ensemble<double>;
using Matrix4fEnsemble = Matrix4Ensemble<float>;
using Matrix4iEnsemble = Matrix4Ensemble<int>;


template<typename Scalar, dim_t Rows>
using VectorEnsemble = MatrixEnsemble<Scalar, Rows, 1>;

template<typename Scalar>
using Vector2Ensemble  = VectorEnsemble<Scalar, 2>;
using Vector2dEnsemble = Vector2Ensemble<double>;
using Vector2fEnsemble = Vector2Ensemble<float>;
using Vector2iEnsemble = Vector2Ensemble<int>;

template<typename Scalar>
using Vector3Ensemble  = VectorEnsemble<Scalar, 3>;
using Vector3dEnsemble = Vector3Ensemble<double>;
using Vector3fEnsemble = Vector3Ensemble<float>;
using Vector3iEnsemble = Vector3Ensemble<int>;

template<typename Scalar>
using Vector4Ensemble  = VectorEnsemble<Scalar, 4>;
using Vector4dEnsemble = Vector4Ensemble<double>;
using Vector4fEnsemble = Vector4Ensemble<float>;
using Vector4iEnsemble = Vector4Ensemble<int>;

template<typename Scalar>
using Vector6Ensemble  = VectorEnsemble<Scalar, 6>;
using Vector6dEnsemble = Vector6Ensemble<double>;
using Vector6fEnsemble = Vector6Ensemble<float>;
using Vector6iEnsemble = Vector6Ensemble<int>;

} // namespace feta2

// Alias feta2 to feta. This can be disabled for compatibility with original
// feta, in the unlikely case you're using both.
#ifndef FETA2_NOT_FETA
namespace feta = feta2;
#endif
