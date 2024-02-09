#include <iostream>

#include <feta2/feta.cuh>

// Instead of pointers, we pass global references to our kernels.
// A `GRef` type is a non-owning reference to a globally-indexed ensemble type
// (which usually lives in device-global memory)
__global__ void doStuff(feta::Matrix3dEnsemble::GRef foo,
    feta::Vector6dEnsemble::GRef bar, feta::DoubleEnsemble::GRef out)
{
    // SampleIndex for this thread
    feta::SampleIndex i(threadIdx.x, blockIdx.x, blockDim.x);

    // Bounds checking on the number of samples
    if (i.global() >= bar.size())
        return;

    // `bar[i]` behaves like an `Eigen::Vector6d`, but is mapped in global
    // memory. Here we read and write these values directly to/from global
    // memory.
    bar[i] += bar[i] + bar[i];

    // Doesn't compile: which sample should the expression be evaluated for?
    // `bar` itself is not a vector type!
    // bar += bar + bar;

    // Once we have extracted a matrix or vector from the ensemble with `[i]`,
    // we can use all of Eigen's usual tricks
    bar[i].tail<3>() += foo[i] * bar[i].head<3>(); // Matrix-vector product

    // Shared memory buffer
    extern __shared__ double smem[];

    // Work references are created on shared memory, and operate on the subset
    // of the sample population assigned to this thread block
    feta::Vector3dEnsemble::WRef workEnsemble(smem, blockDim.x);

    // Per-block work references can be used seamlessly with global references,
    // and the indexing is correctly mapped between the two since `i` is of type
    // `SampleIndex`. This allows for easy global memory coalescing!
    workEnsemble[i] = bar[i].head<3>() + foo[i].col(2);

    // More Eigen trickery -- by moving into "array land", all operations are
    // performed element-wise (unlike in "vector land" where linear algebra
    // logic rules). This is at zero runtime overhead!
    workEnsemble[i].array() /= bar[i].tail<3>().array();

    // A single 3D vector of doubles, allocated on per-thread register memory
    Eigen::Vector3d vec = 3 * Eigen::Vector3d::Ones();

    // Single elements can play with ensembles, just remember to index ensembles
    // where needed!
    vec += workEnsemble[i] + bar[i].tail<3>();

    // Doesn't compile: which sample should the expression be evaluated for?
    // vec += workEnsemble + bar;

    // Results that need to persist outside this kernel must of course be
    // written back to global memory
    out[i] = vec.lpNorm<Eigen::Infinity>();

    // The above is equivalent to:
    // out[i] = vec.cwiseAbs().maxCoeff();
}

int main()
{
    using feta::idx_t;

    std::cout << "Doing some cheesy magic..." << std::endl;

    constexpr idx_t blockSize = 64;
    const idx_t nSamples      = 1e6;

    // A 3x3 matrix of doubles per sample, with values default-initialised to 0
    feta::Matrix3dEnsemble foo(nSamples);

    // 6x1 vectors initialised to a custom value (each component is the same)
    feta::Vector6dEnsemble bar(nSamples, 1.15);

    // Ensemble of scalar values
    feta::DoubleEnsemble out(nSamples);

    // Copy data to device memory
    foo.asyncMemcpyHostToDevice();
    bar.asyncMemcpyHostToDevice();
    out.asyncMemcpyHostToDevice();

    // Number of CUDA blocks to launch
    const idx_t nBlocks = (nSamples + blockSize - 1) / blockSize;

    // Size in bytes of the shared memory buffer required to work on a
    // Vector3dEnsemble
    const idx_t shMem = feta::Vector3dEnsemble::WRef::bufBytes(blockSize);

    // The KERNEL_PRE and KERNEL_POST macros manage error handling, and are
    // empty unless compiling with -DFETA_DEBUG_MODE
    FETA2_KERNEL_PRE();
    doStuff<<<nBlocks, blockSize, shMem>>>(
        foo.deviceRef(), bar.deviceRef(), out.deviceRef());
    // This will throw an exception in debug mode if there was an error during
    // the kernel execution
    FETA2_KERNEL_POST();

    // Get the results back on the host
    out.asyncMemcpyDeviceToHost();

    // Wait for everything to complete
    cudaDeviceSynchronize();

    double cheesiest = 0;
    for (idx_t i = 0; i < nSamples; i++) {
        cheesiest = std::max(cheesiest, out[i]);
    }

    std::cout << "Enjoy your salad with " << cheesiest << " chunks of feta!\n";
}
