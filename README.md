# FGAC
fast astc texture compression using cuda

# Overview
ASTC is a fixed-rate, lossy texture compression system that is designed to offer an unusual degree of flexibility and to support a very wide range of use cases, while providing better image quality than most formats in common use today.

However, ASTC need to search a lot of candidate block modes to find the best format. It takes a lot of time to compress the texture in practice. Since the texture block compression is not relevant at all, we can compress the texture parallelly with GPU.

In our project, we compress the texture with Cuda. The compression algorithm is based on the arm astc-enc implementation. It's a CPU-based compression program. We port arm-astc to GPU and make full use of the cuda to acclerate the texture compression.

A naive implementation of GPU ASTC compression is compressing the ASTC texture block per thread, that is, task parallel compression. Since the block compression task is heavy and uses many registers, the number of active warps is low, which causes low occupancy. To make full use of the GPU, we use data parallel to compress the ASTC block per CUDA block. It splits the "for loop" task into each thread and shares the data between lanes by warp shuffle as possible as we can.

The astc-enc implementation has a large number of intermediate buffers during candidate block mode searching, which has little performance impact on CPU-based implementations, but has a significant impact on GPU-based implementations. We have optimized this algorithm by in-place update, which removes the intermediate buffer.

<p align="left">
    <img src="/resource/blcok_mode_iteration.png" width="60%" height="60%">
</p>

