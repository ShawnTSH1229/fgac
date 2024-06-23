#ifndef _FGAC_COMPRESS_SYMBOLIC_H_
#define _FGAC_COMPRESS_SYMBOLIC_H_
#include "fgac_decvice_common.cuh"
#include "fgac_ideal_endpoints_and_weights.cuh"
#include "fgac_pick_best_endpoint_format.cuh"
#include "fgac_color_quantize.cuh"
#include "fgac_decompress_symbolic.cuh"
#include "fgac_symbolic_physical.cuh"
#include "../common/helper_math.h"

__device__ float compress_symbolic_block_for_partition_1plane(
	const block_size_descriptor& bsd,
	const image_block& blk,
	float tune_errorval_threshold,
	symbolic_compressed_block& scb,
	compression_working_buffers& tmpbuf,
	int quant_limit)
{
	int max_weight_quant = min(static_cast<int>(QUANT_32), quant_limit);

	// Compute ideal weights and endpoint colors, with no quantization or decimation
	endpoints_and_weights& ei = tmpbuf.ei1;
	compute_ideal_colors_and_weights_1plane(blk, ei);

	int8_t* qwt_bitcounts = tmpbuf.qwt_bitcounts;
	float* qwt_errors = tmpbuf.qwt_errors;

	unsigned int max_block_modes = bsd.block_mode_count_1plane_selected;
	for (uint32_t i = 0; i < max_block_modes; i++)
	{
		const block_mode& bm = bsd.block_modes[i];
		if (bm.quant_mode > max_weight_quant)
		{
			qwt_errors[i] = 1e38f;
			continue;
		}

		int bitcount = 115 - 4 - bm.weight_bits;
		if (bitcount <= 0)
		{
			qwt_errors[i] = 1e38f;
			continue;
		}

		qwt_bitcounts[i] = int8_t(bitcount);

		float weights_uquantf[BLOCK_MAX_WEIGHTS];;
		compute_quantized_weights_for_decimation(bsd, ei, weights_uquantf, quant_method(bm.quant_mode));
		qwt_errors[i] = compute_error_of_weight_set_1plane(ei, bsd, weights_uquantf);

#if ASTC_DEBUG_COUT
		std::cout << "Quant Mode:" << quant_metod_str[bm.quant_mode] << " Weight Bits:" << int(bm.weight_bits) << " Quant Error:" << qwt_errors[i] << std::endl;
#endif
	}

	// Decide the optimal combination of color endpoint encodings and weight encodings
	uint8_t best_ep_format_specifiers[TUNE_MAX_TRIAL_CANDIDATES];
	int block_mode_index[TUNE_MAX_TRIAL_CANDIDATES];

	quant_method color_quant_level[TUNE_MAX_TRIAL_CANDIDATES];
	//quant_method color_quant_level_mode[TUNE_MAX_TRIAL_CANDIDATES];

	// Iterate over the N believed-to-be-best modes to find out which one is actually best
	float best_errorval_in_mode = ERROR_CALC_DEFAULT;
	float best_errorval_in_scb = scb.errorval;

	uint32_t candidate_count = compute_ideal_endpoint_formats(
		blk, ei.ep, qwt_bitcounts, qwt_errors,0, 
		max_block_modes, best_ep_format_specifiers, block_mode_index, color_quant_level, tmpbuf);

	// Iterate over the N believed-to-be-best modes to find out which one is actually best
	for (unsigned int i = 0; i < candidate_count; i++)
	{
		const block_mode& qw_bm = bsd.block_modes[block_mode_index[i]];

#if ASTC_DEBUG_COUT
		std::cout << "Candidate Index:" << i << "\t" << "Weight Quant Mode:" << quant_metod_str[qw_bm.quant_mode] << "\t" << " Weight Bits:" << int(qw_bm.weight_bits) << "\t" << "Best Endpoint Format:" << end_point_format_str[best_ep_format_specifiers[i]] << "\t" << "Color Quant Mode:" << quant_metod_str[color_quant_level[i]] << std::endl;
#endif

		// Recompute the ideal color endpoints before storing them
		float4 rgbs_colors;

		symbolic_compressed_block workscb;
		endpoints workep = ei.ep;

		for (unsigned int j = 0; j < blk.texel_count; j++)
		{
			workscb.weights[j] = ei.weights[j] * 64;
		}

		recompute_ideal_colors_1plane(blk, rgbs_colors);

		workscb.color_formats = pack_color_endpoints(
			workep.endpt0,
			workep.endpt1,
			rgbs_colors,
			best_ep_format_specifiers[i],
			workscb.color_values,
			color_quant_level[i]
		);

		// Store header fields
		workscb.quant_mode = color_quant_level[i];
		workscb.block_mode = qw_bm.mode_index;
		workscb.block_type = SYM_BTYPE_NONCONST;

		float errorval = compute_symbolic_block_difference_1plane_1partition(ei, bsd, workscb, blk);
		if (errorval == ERROR_CALC_DEFAULT)
		{
			errorval = -errorval;
			workscb.block_type = SYM_BTYPE_ERROR;
		}

		best_errorval_in_mode = fmin(errorval, best_errorval_in_mode);

		if (errorval < best_errorval_in_scb)
		{
			best_errorval_in_scb = errorval;
			workscb.errorval = errorval;
			scb = workscb;

			if (errorval < tune_errorval_threshold)
			{
				// Skip remaining candidates - this is "good enough"
				i = candidate_count;
				break;
			}
		}
	}

	return best_errorval_in_mode;
}

__device__ void compress_block(fgac_contexti* ctx, image_block* blk, uint8_t pcb[16], compression_working_buffers* tmpbuf)
{
	symbolic_compressed_block scb;
	const block_size_descriptor& bsd = ctx->bsd;
	const fgac_config& config = ctx->config;
	if ((blk->data_min.x == blk->data_max.x) &&
		(blk->data_min.y == blk->data_max.y) &&
		(blk->data_min.z == blk->data_max.z) &&
		(blk->data_min.w == blk->data_max.w))
	{
		scb.block_type = SYM_BTYPE_CONST_U16;
		float4 color_f32 = clamp(blk->data_min, make_float4(0.0), make_float4(65536.0));
		scb.constant_color[0] = int(color_f32.x + 0.5);
		scb.constant_color[1] = int(color_f32.y + 0.5);
		scb.constant_color[2] = int(color_f32.z + 0.5);
		scb.constant_color[3] = int(color_f32.w + 0.5);
		symbolic_to_physical(*blk,bsd, scb, pcb);
	}

	float error_weight_sum = config.cw_sum_weight * bsd.texel_count;
	float error_threshold = config.tune_db_limit * error_weight_sum;

	// Set SCB and mode errors to a very high error value
	scb.errorval = ERROR_CALC_DEFAULT;
	scb.block_type = SYM_BTYPE_ERROR;

	// search 1 partition and 1 plane
	float errorval = compress_symbolic_block_for_partition_1plane(bsd, *blk, error_threshold, scb, *tmpbuf, QUANT_32);

	symbolic_to_physical(*blk, bsd, scb, pcb);
}
#endif