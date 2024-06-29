#include "fgac_device_common.cuh"
#include "fgac_pick_best_endpoint_format.cuh"
#include "fgac_compress_symbolic.cuh"
#include "fgac_color_quantize.cuh"
#include "fgac_decompress_symbolic.cuh"
#include "fgac_symbolic_physical.cuh"
#include "fgac_averages_and_directions.cuh"
#include "fgac_ideal_endpoints_and_weights.cuh"

#if CUDA_OUTBUFFER_DEBUG
#include <stdio.h>
#endif

#define CUDA_VARIFY(expr)\
if (expr != cudaSuccess)\
{\
__debugbreak(); \
}\


#define ASTC_BLOCK_FORMAT_SIZE  4 /*ASTC 4*4 */

__global__ void gpu_encode_kernel(uint8_t * dstData, const uint8_t* const srcData, const block_size_descriptor* const bsd, uint32_t tex_size_x, uint32_t tex_size_y, uint32_t blk_num_x, uint32_t blk_num_y
#if CUDA_OUTBUFFER_DEBUG
	,uint8_t* debug_out_buffer
#endif
)
{
	uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
	uint32_t tid = threadIdx.x;
	uint32_t lane_id = tid % 32;
	uint32_t blk_dt_byte = bid * 16;

	unsigned mask = __ballot_sync(0xFFFFFFFFu, tid < BLOCK_MAX_TEXELS);

	float3 datav;
	uint32_t is_all_same_group = 0;

	if (tid < BLOCK_MAX_TEXELS)
	{
		uint2 blk_start_idx = make_uint2(blockIdx.x * ASTC_BLOCK_FORMAT_SIZE, blockIdx.y * ASTC_BLOCK_FORMAT_SIZE);
		uint2 tex_idx_2d = blk_start_idx + make_uint2(tid % ASTC_BLOCK_FORMAT_SIZE, tid / ASTC_BLOCK_FORMAT_SIZE);
		uint tex_idx = tex_idx_2d.y * tex_size_x + tex_idx_2d.x;

		uint32_t trd_byte_offset = tex_idx * 4;

		const uint8_t* data = &srcData[trd_byte_offset];
		datav = make_float3(data[0], data[1], data[2]) * (65535.0f / 255.0f);
		shared_blk_data[tid] = datav;

		int4 is_all_same;
		__match_all_sync(mask, datav.x, &is_all_same.x);
		__match_all_sync(mask, datav.y, &is_all_same.y);
		__match_all_sync(mask, datav.z, &is_all_same.z);

		if (is_all_same.x && is_all_same.y && is_all_same.z)
		{
			// constant block
			if (tid == 0)
			{
				shared_best_symbolic_block.block_type = SYM_BTYPE_CONST_U16;
				shared_best_symbolic_block.constant_color[0] = int(datav.x + 0.5);
				shared_best_symbolic_block.constant_color[1] = int(datav.y + 0.5);
				shared_best_symbolic_block.constant_color[2] = int(datav.z + 0.5);
				shared_best_symbolic_block.constant_color[3] = int(0);
				symbolic_to_physical(bsd, shared_best_symbolic_block, dstData + blk_dt_byte);
				is_all_same_group = 1;
			}
		}
		__syncwarp(mask);
		is_all_same_group = __shfl_sync(mask, is_all_same_group, 0, 32);
	}
	__syncwarp();

	if (is_all_same_group)
	{
		return;
	}

	if (tid < BLOCK_MAX_TEXELS)
	{
		float3 sum = warp_reduce_vec_sum(mask, datav);
		__syncwarp(mask);
		
		sum = warp_boardcast_vec(mask, sum);
		const float3 data_mean = sum / BLOCK_MAX_TEXELS;

		float3 safe_dir = compute_avgs_and_dirs_3_comp(datav, data_mean, lane_id, mask);
		compute_ideal_colors_and_weights_4_comp(datav, lane_id, tid, mask, data_mean, safe_dir);
		compute_encoding_choice_errors(data_mean, safe_dir, datav, lane_id, tid, mask);
	}

	__syncthreads();
	compute_color_error_for_every_integer_count_and_quant_level(bsd, tid);
	find_candidate_block_modes(bsd, tid);
	
	// Iterate over the N believed-to-be-best modes to find out which one is actually best
	{
		recompute_ideal_colors_1plane(lane_id, tid, mask);

		if (tid == 0)
		{
			shared_block_compress_error = 1e30f;
		}
		
		compute_symbolic_block_difference_1plane_1partition(tid, bsd);
	}
	__syncwarp();

	if (tid == 0)
	{
#if CUDA_OUTBUFFER_DEBUG
		for (int d_idx = 0; d_idx < 16; d_idx++)
		{
			printf("weights: %d %d\n", d_idx, int(shared_best_symbolic_block.weights[d_idx]));
		}
#endif
		symbolic_to_physical(bsd, shared_best_symbolic_block, dstData + blk_dt_byte);
	}
}

extern "C" void image_compress(uint8_t * dstData, const uint8_t* const srcData, const block_size_descriptor* const bsd, uint32_t tex_size_x, uint32_t tex_size_y, uint32_t blk_num_x, uint32_t blk_num_y, uint32_t dest_offset
#if CUDA_OUTBUFFER_DEBUG
	,uint8_t* host_debug_buffer
#endif
)
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

	int bsd_size = sizeof(block_size_descriptor);

	block_size_descriptor* device_bsd;
	CUDA_VARIFY(cudaMalloc((void**)&device_bsd, bsd_size));
	CUDA_VARIFY(cudaMemcpy(device_bsd, bsd, bsd_size, cudaMemcpyHostToDevice));


#if CUDA_OUTBUFFER_DEBUG
	int debug_buffer_size = 4 * 4 * 4;
	uint8_t* debug_out_buffer = nullptr;
	CUDA_VARIFY(cudaMalloc((void**)&debug_out_buffer, debug_buffer_size));
#endif

	gpu_encode_kernel <<< grid_size, block_size >>> (dest_device_data, src_device_data, device_bsd, tex_size_x, tex_size_y, blk_num_x, blk_num_y
#if CUDA_OUTBUFFER_DEBUG
		,debug_out_buffer
#endif
		);

	CUDA_VARIFY(cudaMemcpy(dstData + dest_offset, dest_device_data, dest_astc_size, cudaMemcpyDeviceToHost));
#if CUDA_OUTBUFFER_DEBUG
	CUDA_VARIFY(cudaMemcpy(host_debug_buffer, debug_out_buffer, debug_buffer_size, cudaMemcpyDeviceToHost));
#endif

	cudaFree(src_device_data);
	cudaFree(dest_device_data);
}