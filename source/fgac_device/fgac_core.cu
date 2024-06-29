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

		// compute_ideal_colors_and_weights_4_comp
		compute_ideal_colors_and_weights_4_comp(datav, lane_id, tid, mask, data_mean, safe_dir);

		//compute_encoding_choice_errors
		compute_encoding_choice_errors(data_mean, safe_dir, datav, lane_id, tid, mask);

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
	
	// Iterate over the N believed-to-be-best modes to find out which one is actually best
	{
		//recompute_ideal_colors_1plane
		if (tid < BLOCK_MAX_TEXELS)
		{
			float scale = dot(float3(shared_scale_dir.x, shared_scale_dir.y, shared_scale_dir.z), float3(shared_blk_data[tid].x, shared_blk_data[tid].y, shared_blk_data[tid].z));
			float min_scale = scale;
			float max_scale = scale;
			__syncwarp(mask);
#pragma unroll
			for (int offset = (BLOCK_MAX_TEXELS >> 1); offset > 0; offset >>= 1)
			{
				float other_min_scale = __shfl_xor_sync(mask, min_scale, offset);
				if (min_scale > other_min_scale)
				{
					min_scale = other_min_scale;
				}

				float other_max_scale = __shfl_xor_sync(mask, max_scale, offset);
				if (max_scale < other_max_scale)
				{
					max_scale = other_max_scale;
				}
			}

			if (tid == 0)
			{
				float scalediv = min_scale / fmaxf(max_scale, 1e-10f);
				scalediv = clamp(scalediv, 0.0, 1.0);
				float3 sds = shared_scale_dir * max_scale;
				shared_rgbs_color = make_float4(sds.x, sds.y, sds.z, scalediv);
#if CUDA_OUTBUFFER_DEBUG
				printf("shared_rgbs_color %f,%f,%f,%f\n", shared_rgbs_color.x, shared_rgbs_color.y, shared_rgbs_color.z, shared_rgbs_color.w);
				printf("max_scale % f\n", max_scale);
				printf("min_scale % f\n", min_scale);
#endif
			}
		}

		if (tid == 0)
		{
			shared_block_compress_error = 1e30f;
		}

		
		
#pragma unroll
		for (int candidate_loop_idx = 0; candidate_loop_idx < 2; candidate_loop_idx ++)
		{
			__syncwarp();

			int sub_block_idx = tid / 16;
			int in_block_idx = tid % 16;

			int global_idx = sub_block_idx + candidate_loop_idx * 2;
			const block_mode& qw_bm = bsd->block_modes[candidate_block_mode_index[global_idx]];
			
			uint8_t weight_to_encode = shared_blk_weights[in_block_idx] * 64;

			// pack_color_endpoints
			float4 color0 = make_float4(shared_endpt[0].x, shared_endpt[0].y, shared_endpt[0].z, 0.0);
			float4 color1 = make_float4(shared_endpt[1].x, shared_endpt[1].y, shared_endpt[1].z, 0.0);

			uint8_t color_values[8];
			pack_color_endpoints(color0, color1, shared_rgbs_color, candidate_ep_format_specifiers[global_idx], color_values, quant_method(candidate_color_quant_level[global_idx]));

			int4 ep0;
			int4 ep1;
			bool rgb_lns;
			bool a_lns;

			unpack_color_endpoints(candidate_ep_format_specifiers[global_idx], color_values, ep0, ep1);

			float sum_error = 0.0;
			// compute_symbolic_block_difference_1plane_1partition
			{
				// Compute EP1 contribution
				int weight1 = shared_blk_weights[in_block_idx] * 64.0;
				ep1 = ep1 * weight1;

				// Compute EP0 contribution
				int weight0 = int(64) - weight1;
				ep0 = ep0 * weight0;

				// Combine contributions
				int colori_r = int(ep0.x + ep1.x + int(32)) >> 6;
				int colori_g = int(ep0.y + ep1.y + int(32)) >> 6;
				int colori_b = int(ep0.z + ep1.z + int(32)) >> 6;
				int colori_a = int(ep0.w + ep1.w + int(32)) >> 6;

				// Compute color diff
				float4 colorf = make_float4(colori_r, colori_g, colori_b, 0);
				float4 color_origin = make_float4(shared_blk_data[in_block_idx].x, shared_blk_data[in_block_idx].y, shared_blk_data[in_block_idx].z, 0);

				float4 color_error = make_float4(
					fmin(abs(color_origin.x - colorf.x), float(1e15f)),
					fmin(abs(color_origin.y - colorf.y), float(1e15f)),
					fmin(abs(color_origin.z - colorf.z), float(1e15f)),
					fmin(abs(color_origin.w - colorf.w), float(1e15f)));

				// Compute squared error metric
				color_error = color_error * color_error;

				float metric = color_error.x + color_error.y + color_error.z + color_error.w;
				sum_error = metric;
				__syncwarp();

				sum_error += __shfl_down_sync(0xFFFFFFFF, sum_error, 8, 32);
				sum_error += __shfl_down_sync(0xFFFFFFFF, sum_error, 4, 32);
				sum_error += __shfl_down_sync(0xFFFFFFFF, sum_error, 2, 32);
				sum_error += __shfl_down_sync(0xFFFFFFFF, sum_error, 1, 32);

#if CUDA_OUTBUFFER_DEBUG
				if (candidate_loop_idx == 0 && sub_block_idx == 0)
				{
					printf("tid: %d , original color z: %f, colorfz:%d, metric:%f\n", tid, color_origin.z, int(color_values[1]), metric);
					printf("tid: %d , color_error x: %f, color_error y:%f, color_error z:%f\n", tid, color_error.x, color_error.y, color_error.z);
				}
#endif
				__syncwarp();
				int min_tid = sub_block_idx * 16;
				float other_sum_error = __shfl_xor_sync(0x00010001, sum_error, 16);
				float other_tid = __shfl_xor_sync(0x00010001, tid, 16);
				if (other_sum_error < sum_error || (other_sum_error == sum_error && other_tid < tid))
				{
					min_tid = (16 - min_tid);
				}
				__syncwarp();
				if (tid >= min_tid && tid < (min_tid + 16))
				{
					if (sum_error < shared_block_compress_error)
					{
						int tid_offset = tid - min_tid;
						if (tid_offset < 8)
						{
							shared_best_symbolic_block.color_values[tid_offset] = color_values[tid_offset];
						}

#if CUDA_OUTBUFFER_DEBUG
						printf("debug %d, %d\n", tid_offset, int(weight_to_encode));
#endif
						shared_best_symbolic_block.weights[tid_offset] = weight_to_encode;

						if (tid_offset == 0)
						{
							shared_best_symbolic_block.quant_mode = quant_method(candidate_color_quant_level[global_idx]);
							shared_best_symbolic_block.block_mode = qw_bm.mode_index;;
							shared_best_symbolic_block.block_type = SYM_BTYPE_NONCONST;
							shared_best_symbolic_block.color_formats = candidate_ep_format_specifiers[global_idx];
						}

						shared_block_compress_error = sum_error;
					}
				}
			}
		}
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