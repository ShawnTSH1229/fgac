#ifndef _FGAC_COMPRESS_SYMBOLIC_H_
#define _FGAC_COMPRESS_SYMBOLIC_H_
#include "fgac_device_common.cuh"
#include "fgac_weight_quant_xfer_tables.cuh"
#include "fgac_quantization.cuh"

#if CUDA_OUTBUFFER_DEBUG
#include <stdio.h>
#endif

// The available quant levels, stored with a minus 1 bias
__constant__ float quant_levels_m1[12]{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f, 9.0f, 11.0f, 15.0f, 19.0f, 23.0f, 31.0f };

__inline__ __device__ void compress_block(const block_size_descriptor* const bsd, uint32_t tid)
{


	if (tid < 4)
	{
		candidate_combine_errors[tid] = 1e30f;
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


			// Look up the two closest indexes and return the one that was closest(quant)
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

				if (in_block_idx == 0)
				{

					const uint32_t min_combine_error_mask1 = 0x00010001;

					other_integer_count_error = __shfl_xor_sync(min_combine_error_mask1, integer_count_error, 1);
					int other_best_integer_count = __shfl_xor_sync(min_combine_error_mask1, best_integer_count, 1);
					if (other_integer_count_error < integer_count_error)
					{
						best_integer_count = other_best_integer_count;
						integer_count_error = other_integer_count_error;
					}

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

				#pragma unroll
				for (int candiate_idx = 0; candiate_idx < 6; candiate_idx++)
				{
					if (candidate_combine_errors[candiate_idx] < current_tid_error)
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
		}

		block_mode_process_idx += 2;
	}

	__syncwarp();

#if CUDA_OUTBUFFER_DEBUG
	if (tid < 4)
	{
		printf("candidate quant level: %d, candidate ep format: %d, error: %f\n", candidate_color_quant_level[tid], candidate_ep_format_specifiers[tid], candidate_combine_errors[tid]);
	}
#endif
}

#endif