#include "fgac_device_common.cuh"

#define CUDA_VARIFY(expr)\
if (expr != cudaSuccess)\
{\
__debugbreak(); \
}\

template <typename T>
__inline__ __device__ T warp_reduce_sum(unsigned mask ,T val) 
{
#pragma unroll
	for (int offset = (BLOCK_MAX_TEXELS >> 1); offset > 0; offset >>= 1) 
	{
		val += __shfl_down_sync(mask, val, offset, warpSize);
	}
	return val;
}

template <typename T>
__inline__ __device__ T warp_reduce_vec_sum(unsigned mask, T val) 
{
	T ret;
	ret.x = warp_reduce_sum(mask, val.x);
	ret.y = warp_reduce_sum(mask, val.y);
	ret.z = warp_reduce_sum(mask, val.z);
	return ret;
}

__global__ void gpu_encode_kernel(uint8_t * dstData, const uint8_t* const srcData, const block_size_descriptor* const bsd, uint32_t tex_size_x, uint32_t tex_size_y, uint32_t blk_num_x, uint32_t blk_num_y)
{
	uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
	uint32_t blk_st_texel = bid * blockDim.x * blockDim.y;
	uint32_t blk_st_byte = blk_st_texel * 4;

	uint32_t tid = threadIdx.x;
	unsigned mask = __ballot_sync(0x0000FFFFu, tid < BLOCK_MAX_TEXELS);
	if (tid < BLOCK_MAX_TEXELS)
	{
		uint32_t trd_byte_offset = blk_st_byte + tid * 4;

		const uint8_t* data = &srcData[trd_byte_offset];
		float3 datav = make_float3(data[0], data[1], data[2]) * (65535.0f / 255.0f);

		int4 is_all_same;
		__match_all_sync(mask, datav.x, &is_all_same.x);
		__match_all_sync(mask, datav.y, &is_all_same.y);
		__match_all_sync(mask, datav.z, &is_all_same.z);

		if (is_all_same.x && is_all_same.y && is_all_same.z)
		{
			// constant block
		}

		float3 sum = warp_reduce_vec_sum(mask, datav);

		//todo boardcast
		float4 data_mean = make_float4(0,0,0,0);
	}


	



	float4 mean = { 0,0,0,0 };

	//block_src_tex[tid] = datav;



	uint32_t blk_dt_byte = bid * 16;
	dstData[blk_dt_byte] = srcData[blk_st_byte];
}

extern "C" void image_compress(uint8_t * dstData, const uint8_t* const srcData, const block_size_descriptor* const bsd, uint32_t tex_size_x, uint32_t tex_size_y, uint32_t blk_num_x, uint32_t blk_num_y, uint32_t dest_offset)
{
	dim3 grid_size(blk_num_x, blk_num_y);
	dim3 block_size(32, 1); // [warp size, 1]

	uint32_t src_data_size = tex_size_x * tex_size_y * 4;
	uint8_t* src_device_data = nullptr;
	CUDA_VARIFY(cudaMalloc((void**)&src_device_data, src_data_size));
	CUDA_VARIFY(cudaMemcpy(src_device_data, srcData, src_data_size, cudaMemcpyHostToDevice));

	uint8_t* dest_device_data = nullptr;
	uint32_t dest_astc_size = blk_num_x * blk_num_y * 16;
	CUDA_VARIFY(cudaMalloc((void**)&dest_device_data, dest_astc_size));

	block_size_descriptor* device_bsd;
	CUDA_VARIFY(cudaMalloc((void**)&device_bsd, sizeof(bsd)));
	CUDA_VARIFY(cudaMemcpy(device_bsd, bsd, sizeof(bsd), cudaMemcpyHostToDevice));

	gpu_encode_kernel <<< grid_size, block_size >>> (dest_device_data, src_device_data, device_bsd, tex_size_x, tex_size_y, blk_num_x, blk_num_y);

	CUDA_VARIFY(cudaMemcpy(dstData + dest_offset, dest_device_data, dest_astc_size, cudaMemcpyDeviceToHost));

	cudaFree(src_device_data);
	cudaFree(dest_device_data);
}