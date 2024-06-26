#include "fgac_device_common.cuh"
#include "fgac_pick_best_endpoint_format.cuh"
#include "fgac_compress_symbolic.cuh"

#if CUDA_OUTBUFFER_DEBUG
#include <stdio.h>
#endif

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

__inline__ __device__ float3 warp_boardcast_vec(unsigned mask, float3 val)
{
	float3 ret;
	ret.x = __shfl_sync(mask, val.x, 0, BLOCK_MAX_TEXELS);
	ret.y = __shfl_sync(mask, val.y, 0, BLOCK_MAX_TEXELS);
	ret.z = __shfl_sync(mask, val.z, 0, BLOCK_MAX_TEXELS);
	return ret;
}

__global__ void gpu_encode_kernel(uint8_t * dstData, const uint8_t* const srcData, const block_size_descriptor* const bsd, uint32_t tex_size_x, uint32_t tex_size_y, uint32_t blk_num_x, uint32_t blk_num_y
#if CUDA_OUTBUFFER_DEBUG
	,uint8_t* debug_out_buffer
#endif
)
{
	uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
	uint32_t blk_st_texel = bid * blockDim.x * blockDim.y;
	uint32_t blk_st_byte = blk_st_texel * 4;

	uint32_t tid = threadIdx.x;
	uint32_t lane_id = tid % 32;

#if CUDA_OUTBUFFER_DEBUG
	__syncwarp();
	if (tid == 0)
	{
		printf("debug error: %d\n", int(bsd->block_modes[1].mode_index));
	}
#endif

	unsigned mask = __ballot_sync(0xFFFFFFFFu, tid < BLOCK_MAX_TEXELS);
	if (tid < BLOCK_MAX_TEXELS)
	{
		uint32_t trd_byte_offset = blk_st_byte + tid * 4;

		const uint8_t* data = &srcData[trd_byte_offset];
		const float3 datav = make_float3(data[0], data[1], data[2]) * (65535.0f / 255.0f);
		shared_blk_data[bid] = datav;

		int4 is_all_same;
		__match_all_sync(mask, datav.x, &is_all_same.x);
		__match_all_sync(mask, datav.y, &is_all_same.y);
		__match_all_sync(mask, datav.z, &is_all_same.z);

		if (is_all_same.x && is_all_same.y && is_all_same.z)
		{
			// constant block
		}

		float3 sum = warp_reduce_vec_sum(mask, datav);
		__syncwarp(mask);
		
		sum = warp_boardcast_vec(mask, sum);
		
		const float3 data_mean = sum / BLOCK_MAX_TEXELS;

		float3 safe_dir = float3(0, 0, 0);

		// compute_avgs_and_dirs_3_comp
		{
			float3 texel_datum = datav - data_mean;

			unsigned sum_x_mask = __ballot_sync(mask, texel_datum.x > 0);
			unsigned sum_y_mask = __ballot_sync(mask, texel_datum.y > 0);
			unsigned sum_z_mask = __ballot_sync(mask, texel_datum.z > 0);

			float3 valid_sum_xp = (sum_x_mask & (1 << lane_id)) != 0 ? texel_datum : float3(0, 0, 0);
			float3 valid_sum_yp = (sum_y_mask & (1 << lane_id)) != 0 ? texel_datum : float3(0, 0, 0);
			float3 valid_sum_zp = (sum_z_mask & (1 << lane_id)) != 0 ? texel_datum : float3(0, 0, 0);

			__syncwarp(mask);

			float3 sum_xp = warp_reduce_vec_sum(mask, valid_sum_xp);
			float3 sum_yp = warp_reduce_vec_sum(mask, valid_sum_yp);
			float3 sum_zp = warp_reduce_vec_sum(mask, valid_sum_zp);

			if (lane_id == 0)
			{
				float prod_xp = dot(sum_xp, sum_xp);
				float prod_yp = dot(sum_yp, sum_yp);
				float prod_zp = dot(sum_zp, sum_zp);

				float3 best_vector = sum_xp;
				float best_sum = prod_xp;

				if (prod_yp > best_sum)
				{
					best_vector = sum_yp;
					best_sum = prod_yp;
				}

				if (prod_zp > best_sum)
				{
					best_vector = sum_zp;
					best_sum = prod_zp;
				}

				if ((best_vector.x + best_vector.y + best_vector.z) < 0.0f)
				{
					best_vector = -best_vector;
				}

				float length_dir = length(best_vector);
				safe_dir = (length_dir < 1e-10) ? normalize(make_float3(1.0)) : normalize(best_vector);
			}
		}

		// compute_ideal_colors_and_weights_4_comp
		{
			__syncwarp(mask);
			safe_dir = warp_boardcast_vec(mask, safe_dir);
			const float param = dot(datav - data_mean, safe_dir);
			float lowparam = param;
			float highparam = param;

			__syncwarp(mask);
#pragma unroll
			for (int offset = (BLOCK_MAX_TEXELS >> 1); offset > 0; offset >>= 1)
			{
				float other_low_param = __shfl_xor_sync(mask, lowparam, offset);
				if (lowparam > other_low_param)
				{
					lowparam = other_low_param;
				}

				float other_highparam = __shfl_xor_sync(mask, highparam, offset);
				if (highparam < other_highparam)
				{
					highparam = other_highparam;
				}
			}

			float length;
			float scale;

			if (lane_id == 0)
			{
				if (highparam <= lowparam)
				{
					lowparam = 0.0f;
					highparam = 1e-7f;
				}

				float3 endpt0 = data_mean + safe_dir * lowparam;
				float3 endpt1 = data_mean + safe_dir * highparam;

				length = highparam - lowparam;
				scale = 1.0f / length;
				shared_endpt[0] = endpt0;
				shared_endpt[1] = endpt1;
			}

			__syncwarp(mask);

			length = __shfl_sync(mask, length, 0, BLOCK_MAX_TEXELS);
			scale = __shfl_sync(mask, scale, 0, BLOCK_MAX_TEXELS);

			float idx = (param - lowparam) * scale;
			idx = clamp(idx, 0.0, 1.0);
			float weight = idx;

			shared_blk_weights[tid] = weight;
		}

		//compute_encoding_choice_errors
		{
			line3 uncor_rgb_lines;
			line3 samec_rgb_lines;

			processed_line3 uncor_rgb_plines;
			processed_line3 samec_rgb_plines;
			processed_line3 luminance_plines;

			uncor_rgb_lines.a = data_mean;
			uncor_rgb_lines.b = safe_dir;

			samec_rgb_lines.a = make_float3(0);
			samec_rgb_lines.b = normalize_safe(data_mean);

			uncor_rgb_plines.amod = uncor_rgb_lines.a - uncor_rgb_lines.b * dot(uncor_rgb_lines.a, uncor_rgb_lines.b);
			uncor_rgb_plines.bs = uncor_rgb_lines.b;

			// Same chroma always goes though zero, so this is simpler than the others
			samec_rgb_plines.amod = make_float3(0);
			samec_rgb_plines.bs = samec_rgb_lines.b;

#if CUDA_OUTBUFFER_DEBUG
			if (lane_id == 0)
			{
				printf("lane_id %d, x: %f,x: %f,x: %f\n", lane_id, data_mean.x, data_mean.y, data_mean.z);
			}
#endif

			// Luminance always goes though zero, so this is simpler than the others
			float val = 0.577350258827209473f;
			luminance_plines.amod = make_float3(0);
			luminance_plines.bs = make_float3(val, val, val);

			// Compute uncorrelated error
			float param = dot(datav, uncor_rgb_plines.bs);
			float3 dist = (uncor_rgb_plines.amod + param * uncor_rgb_plines.bs) - datav;
			float uncor_err = dot(dist, dist);

			// Compute same chroma error - no "amod", its always zero
			param = dot(datav, samec_rgb_plines.bs);
			dist = param * samec_rgb_plines.bs - datav;
			float samec_err = dot(dist, dist);

			// Compute luma error - no "amod", its always zero
			param = dot(datav, luminance_plines.bs);
			dist = param * luminance_plines.bs - datav;
			float l_err = dot(dist, dist);

			__syncwarp(mask);

			float sum_uncor_err = warp_reduce_sum(mask, uncor_err);
			float sum_samec_err = warp_reduce_sum(mask, samec_err);
			float sum_l_err = warp_reduce_sum(mask, l_err);

			if (lane_id == 0)
			{
				shared_rgb_scale_error = (sum_samec_err - sum_uncor_err) * 0.7f;// empirical
				shared_luminance_error = (sum_l_err - sum_uncor_err) * 3.0f;// empirical
			}
		}

#if CUDA_OUTBUFFER_DEBUG
		
		debug_out_buffer[tid * 4 + 0] = 255;
		debug_out_buffer[tid * 4 + 1] = 255;
		debug_out_buffer[tid * 4 + 2] = 255;
		debug_out_buffer[tid * 4 + 3] = 255;
#endif
	}

	__syncthreads();
	compute_color_error_for_every_integer_count_and_quant_level(bsd, tid);

	compress_block(bsd, tid);
	

	uint32_t blk_dt_byte = bid * 16;
	dstData[blk_dt_byte] = srcData[blk_st_byte];
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

	block_size_descriptor* device_bsd;
	CUDA_VARIFY(cudaMalloc((void**)&device_bsd, sizeof(bsd)));
	CUDA_VARIFY(cudaMemcpy(device_bsd, bsd, sizeof(bsd), cudaMemcpyHostToDevice));

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