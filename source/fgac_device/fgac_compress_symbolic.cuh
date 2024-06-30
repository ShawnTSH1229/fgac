#ifndef _FGAC_COMPRESS_SYMBOLIC_H_
#define _FGAC_COMPRESS_SYMBOLIC_H_
#include "fgac_device_common.cuh"
#include "fgac_weight_quant_xfer_tables.cuh"
#include "fgac_quantization.cuh"
#include "fgac_color_quantize.cuh"
#include "fgac_decompress_symbolic.cuh"

#if CUDA_OUTBUFFER_DEBUG
#include <stdio.h>
#endif

// The available quant levels, stored with a minus 1 bias
__constant__ float quant_levels_m1[12]{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f, 9.0f, 11.0f, 15.0f, 19.0f, 23.0f, 31.0f };

__inline__ __device__ void find_candidate_block_modes(const block_size_descriptor* const bsd, uint32_t tid)
{
	if (tid < 4)
	{
		candidate_combine_errors[tid] = 1e30f;
		candidate_block_mode_index[tid] = 2049; //invalid
	}

	//** compute_ideal_endpoint_formats
	unsigned int max_block_modes = bsd->block_mode_count_1plane_selected;
	int block_mode_process_idx = 0;
	while (block_mode_process_idx < max_block_modes)
	{
		__syncwarp();

		int sub_block_idx = tid / 16;
		int in_block_idx = tid % 16;

		int global_idx = sub_block_idx + block_mode_process_idx; // ignore the last block mode for now
		bool is_block_mode_index_valid = (block_mode_process_idx + 1) < max_block_modes;
		if (is_block_mode_index_valid)
		{
			const block_mode bm = bsd->block_modes[global_idx];
			int bitcount = 115 - 4 - bm.weight_bits;

			quant_method quant_level = quant_method(bm.quant_mode);

			// ** compute_quantized_weights_for_decimation
			const quant_and_transfer_table& qat = quant_and_xfer_tables[quant_level];
			uint quant_level_map = get_quant_level(quant_level);
			int steps_m1(quant_level_map - 1);
			float quant_level_m1 = quant_levels_m1[quant_level];
			float rscale = 1.0f / 64.0f;

			float ix = shared_blk_weights[in_block_idx];
			ix = clamp(ix, 0.0, 1.0);
			float eai_weight = ix;


			// ** Look up the two closest indexes and return the one that was closest(quant)
			float ix1 = ix * quant_level_m1;

			int weightl = int(ix1);
			int weighth = min(weightl + 1, steps_m1);

			float ixl = qat.quant_to_unquant[weightl];
			float ixh = qat.quant_to_unquant[weighth];

			if (ixl + ixh < 128.0f * ix)
			{
				ixl = ixh;
			}
			else
			{
				ixl = ixl;
			}

			// Invert the weight-scaling that was done initially
			float weight_quant_uvalue = ixl * rscale;

			//** compute_error_of_weight_set_1plane

			// Load the weight set directly, without interpolation
			float current_values = weight_quant_uvalue;

			// Compute the error between the computed value and the ideal weight
			float actual_values = eai_weight;
			float diff = current_values - actual_values;
			float error = diff * diff;

			error += __shfl_down_sync(0xFFFFFFFF, error, 8, 32);
			error += __shfl_down_sync(0xFFFFFFFF, error, 4, 32);
			error += __shfl_down_sync(0xFFFFFFFF, error, 2, 32);
			error += __shfl_down_sync(0xFFFFFFFF, error, 1, 32);

			//** one_partition_find_best_combination_for_bitcount
			if (in_block_idx < 4)
			{
				int integer_count = in_block_idx + 1;

				int best_integer_count = integer_count - 1;
				float best_integer_count_error = 1e30f;

				// Compute the quantization level for a given number of integers and a given number of bits
				int quant_level = quant_mode_table[integer_count][bitcount];

				float integer_count_error = 1e30f; // set to the maximum error for quant_level < QUANT_6
				// Don't have enough bits to represent a given endpoint format at all!
				if (quant_level >= QUANT_6)
				{
					integer_count_error = shared_best_error[quant_level][integer_count - 1]; // best combined error
				}

				const uint32_t min_combine_error_mask0 = 0x000F000F;

				__syncwarp(min_combine_error_mask0);

				float other_integer_count_error = __shfl_xor_sync(min_combine_error_mask0, integer_count_error, 2);
				if (other_integer_count_error < integer_count_error)
				{
					integer_count_error = other_integer_count_error;
					best_integer_count = best_integer_count + 2;
				}

				__syncwarp(min_combine_error_mask0);

				const uint32_t min_combine_error_mask1 = 0x000F000F;
				other_integer_count_error = __shfl_xor_sync(min_combine_error_mask1, integer_count_error, 1);
				int other_best_integer_count = __shfl_xor_sync(min_combine_error_mask1, best_integer_count, 1);
				if (other_integer_count_error < integer_count_error)
				{
					best_integer_count = other_best_integer_count;
					integer_count_error = other_integer_count_error;
				}

				if (in_block_idx == 0)
				{
					best_integer_count_error = integer_count_error;

					int ql = quant_mode_table[best_integer_count + 1][bitcount];

					uint8_t best_quant_level = static_cast<uint8_t>(ql);
					uint8_t best_format = FMT_LUMINANCE;

					if (ql >= QUANT_6)
					{
						best_format = shared_format_of_choice[ql][best_integer_count];
					}

					float total_error = best_integer_count_error + error;

					candidate_ep_format_specifiers[4 + sub_block_idx] = best_format;
					candidate_block_mode_index[4 + sub_block_idx] = global_idx;
					candidate_color_quant_level[4 + sub_block_idx] = ql;
					candidate_combine_errors[4 + sub_block_idx] = total_error;
				}

			}
			__syncwarp();

			// simple gpu sort
			if (tid < 6)
			{
				int num_samller = 0;
				float current_tid_error = candidate_combine_errors[tid];
				float current_ep_format_specifier = candidate_ep_format_specifiers[tid];
				int current_blk_mode_idx = candidate_block_mode_index[tid];
				int current_col_quant_level = candidate_color_quant_level[tid];

#if CUDA_OUTBUFFER_DEBUG
				printf("global index:%d\n", current_blk_mode_idx);
#endif

				#pragma unroll
				for (int candiate_idx = 0; candiate_idx < 6; candiate_idx++)
				{
					float other_candidate_error = candidate_combine_errors[candiate_idx];
					if ((other_candidate_error < current_tid_error) || ((other_candidate_error == current_tid_error) && (candiate_idx < tid)))
					{
						num_samller++;
					}
				}

				// 0011 1111
				__syncwarp(0x0000003F);
				candidate_combine_errors[num_samller] = current_tid_error;
				candidate_ep_format_specifiers[num_samller] = current_ep_format_specifier;
				candidate_block_mode_index[num_samller] = current_blk_mode_idx;
				candidate_color_quant_level[num_samller] = current_col_quant_level;
			}
#if CUDA_OUTBUFFER_DEBUG
			if (tid == 0)
			{
				printf("\n");
			}
#endif
		}

		block_mode_process_idx += 2;
	}

	__syncwarp();

#if CUDA_OUTBUFFER_DEBUG
	if (tid < 4)
	{
		printf("candidate quant level: %d, candidate ep format: %d, error: %f, candidate_block_mode_index:%d\n", candidate_color_quant_level[tid], candidate_ep_format_specifiers[tid], candidate_combine_errors[tid], candidate_block_mode_index[tid]);
	}
#endif
}


__inline__ __device__ void compute_symbolic_block_difference_1plane_1partition(const uint32_t& tid, const block_size_descriptor* const bsd)
{
#pragma unroll
	for (int candidate_loop_idx = 0; candidate_loop_idx < 2; candidate_loop_idx++)
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
#endif