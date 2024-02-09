# feta2.cuh
FETA2 brings Eigen to the GPU without loss of performance.

It provides types representing ensembles of Eigen matrices/vectors.
For instance, a `feta2::Vector3dEnsemble` behaves a lot like an
`std::vector<Eigen::Vector3d>`, but the performance will be significantly better,
and makes memory management on host, device global and device shared memory a breeze.

See `feta2_demo.cu` for a quick overview of FETA2's functionality.

