#ifndef _FGAC_PICK_BEST_ENDPOINT_FORMAT_CUH_
#define _FGAC_PICK_BEST_ENDPOINT_FORMAT_CUH_

#include "../common/helper_math.h"
#include "fgac_decvice_common.cuh"
#include "fgac_averages_and_directions.cuh"
#include "fgac_quantization.cuh"

__constant__ float baseline_quant_error[21 - QUANT_6]{
	(65536.0f * 65536.0f / 18.0f) / (5 * 5),
	(65536.0f * 65536.0f / 18.0f) / (7 * 7),
	(65536.0f * 65536.0f / 18.0f) / (9 * 9),
	(65536.0f * 65536.0f / 18.0f) / (11 * 11),
	(65536.0f * 65536.0f / 18.0f) / (15 * 15),
	(65536.0f * 65536.0f / 18.0f) / (19 * 19),
	(65536.0f * 65536.0f / 18.0f) / (23 * 23),
	(65536.0f * 65536.0f / 18.0f) / (31 * 31),
	(65536.0f * 65536.0f / 18.0f) / (39 * 39),
	(65536.0f * 65536.0f / 18.0f) / (47 * 47),
	(65536.0f * 65536.0f / 18.0f) / (63 * 63),
	(65536.0f * 65536.0f / 18.0f) / (79 * 79),
	(65536.0f * 65536.0f / 18.0f) / (95 * 95),
	(65536.0f * 65536.0f / 18.0f) / (127 * 127),
	(65536.0f * 65536.0f / 18.0f) / (159 * 159),
	(65536.0f * 65536.0f / 18.0f) / (191 * 191),
	(65536.0f * 65536.0f / 18.0f) / (255 * 255)
};

__device__ void compute_error_squared_rgb_single_partition(
	const image_block& blk,
	const processed_line3& uncor_pline,
	float& uncor_err,
	const processed_line3& samec_pline,
	float& samec_err,
	const processed_line3& l_pline,
	float& l_err,
	float& a_drop_err)
{
	float default_a(0xFFFF);

	float uncor_bs0(uncor_pline.bs.x);
	float uncor_bs1(uncor_pline.bs.y);
	float uncor_bs2(uncor_pline.bs.z);

	float uncor_amod0(uncor_pline.amod.x);
	float uncor_amod1(uncor_pline.amod.y);
	float uncor_amod2(uncor_pline.amod.z);

	float samec_bs0(samec_pline.bs.x);
	float samec_bs1(samec_pline.bs.y);
	float samec_bs2(samec_pline.bs.z);

	float l_bs0(l_pline.bs.x);
	float l_bs1(l_pline.bs.y);
	float l_bs2(l_pline.bs.z);

	unsigned int texel_count = blk.texel_count;
	for (unsigned int i = 0; i < texel_count; i++)
	{
		// Compute the error that arises from just ditching alpha
		float data_a = blk.data_a[i];
		float alpha_diff = data_a - default_a;
		alpha_diff = alpha_diff * alpha_diff;

		a_drop_err += alpha_diff;

		float data_r = blk.data_r[i];
		float data_g = blk.data_g[i];
		float data_b = blk.data_b[i];

		// Compute uncorrelated error
		float param = data_r * uncor_bs0 + data_g * uncor_bs1 + data_b * uncor_bs2;

		float dist0 = (uncor_amod0 + param * uncor_bs0) - data_r;
		float dist1 = (uncor_amod1 + param * uncor_bs1) - data_g;
		float dist2 = (uncor_amod2 + param * uncor_bs2) - data_b;

		float error = dist0 * dist0 + dist1 * dist1 + dist2 * dist2;
		uncor_err += error;

		// Compute same chroma error - no "amod", its always zero
		param = data_r * samec_bs0+ data_g * samec_bs1 + data_b * samec_bs2;

		dist0 = (param * samec_bs0) - data_r;
		dist1 = (param * samec_bs1) - data_g;
		dist2 = (param * samec_bs2) - data_b;

		error = dist0 * dist0 + dist1 * dist1 + dist2 * dist2;

		samec_err += error;

		// Compute luma error - no "amod", its always zero
		param = data_r * l_bs0
			+ data_g * l_bs1
			+ data_b * l_bs2;

		dist0 = (param * l_bs0) - data_r;
		dist1 = (param * l_bs1) - data_g;
		dist2 = (param * l_bs2) - data_b;

		error = dist0 * dist0 + dist1 * dist1 + dist2 * dist2;

		l_err += error;
	}
};

__device__ void compute_encoding_choice_errors(
	const image_block& blk,
	const endpoints& ep,
	encoding_choice_errors& eci)
{
	float4 dir;
	float4 avg = blk.data_mean;
	avg.w = 0.0;
	compute_avgs_and_dirs_3_comp(blk, dir);

	line3 uncor_rgb_lines;
	line3 samec_rgb_lines;  // for LDR-RGB-scale

	processed_line3 uncor_rgb_plines;
	processed_line3 samec_rgb_plines;
	processed_line3 luminance_plines;

	uncor_rgb_lines.a = avg;
	uncor_rgb_lines.b = normalize_safe(dir);

	samec_rgb_lines.a = make_float4(0);
	samec_rgb_lines.b = normalize_safe(avg);

	uncor_rgb_plines.amod = uncor_rgb_lines.a - uncor_rgb_lines.b * dot(uncor_rgb_lines.a, uncor_rgb_lines.b);
	uncor_rgb_plines.bs = uncor_rgb_lines.b;

	// Same chroma always goes though zero, so this is simpler than the others
	samec_rgb_plines.amod = make_float4(0);
	samec_rgb_plines.bs = samec_rgb_lines.b;

	float val = 0.577350258827209473f;
	// Luminance always goes though zero, so this is simpler than the others
	luminance_plines.amod = make_float4(0);
	luminance_plines.bs = make_float4(val, val, val, 0.0f);

	float uncorr_rgb_error = 0;
	float samechroma_rgb_error = 0;
	float luminance_rgb_error = 0;
	float alpha_drop_error = 0;
	compute_error_squared_rgb_single_partition(blk, uncor_rgb_plines, uncorr_rgb_error, samec_rgb_plines, samechroma_rgb_error, luminance_plines, luminance_rgb_error, alpha_drop_error);

	float4 endpt0 = ep.endpt0;
	float4 endpt1 = ep.endpt1;
	float4 endpt_diff = fabs(endpt1 - endpt0);


	// Store out the settings
	eci.rgb_scale_error = (samechroma_rgb_error - uncorr_rgb_error) * 0.7f;  // empirical
	eci.luminance_error = (luminance_rgb_error - uncorr_rgb_error) * 3.0f;   // empirical
	eci.alpha_drop_error = alpha_drop_error * 3.0f;
}

#if ASTC_DEBUG_COUT

#endif

__device__ void compute_color_error_for_every_integer_count_and_quant_level(
	const image_block& blk,
	const encoding_choice_errors& eci,
	const endpoints& ep,
	float best_error[21][4],
	uint8_t format_of_choice[21][4]
	)
{
	float4 ep0 = ep.endpt0;
	float4 ep1 = ep.endpt1;

	float4 offset(65535.0f, 65535.0f, 65535.0f, 65535.0f);

	float3 error_weight_rgb(1.0, 1.0, 1.0);

	for (int i = QUANT_2; i < QUANT_6; i++)
	{
		best_error[i][3] = ERROR_CALC_DEFAULT;
		best_error[i][2] = ERROR_CALC_DEFAULT;
		best_error[i][1] = ERROR_CALC_DEFAULT;
		best_error[i][0] = ERROR_CALC_DEFAULT;

		format_of_choice[i][3] = FMT_RGBA;
		format_of_choice[i][2] = FMT_RGB;
		format_of_choice[i][1] = FMT_RGB_SCALE;
		format_of_choice[i][0] = FMT_LUMINANCE;
	}

	float base_quant_error_rgb = 3 * blk.texel_count;
	float base_quant_error_a = 1 * blk.texel_count;
	float base_quant_error_rgba = base_quant_error_rgb + base_quant_error_a;


	// Pick among the available LDR endpoint modes
	for (int i = QUANT_6; i <= QUANT_256; i++)
	{

		float base_quant_error = baseline_quant_error[i - QUANT_6];
		float quant_error_rgb = base_quant_error_rgb * base_quant_error;
		float quant_error_rgba = base_quant_error_rgba * base_quant_error;

		// 8 integers can encode as RGBA+RGBA
		float full_ldr_rgba_error = quant_error_rgba;

		best_error[i][3] = full_ldr_rgba_error;
		format_of_choice[i][3] = FMT_RGBA;

		// 6 integers can encode as RGB+RGB or RGBS+AA
		float full_ldr_rgb_error = quant_error_rgb + eci.alpha_drop_error;

		float rgbs_alpha_error = quant_error_rgba + eci.rgb_scale_error;

		if (rgbs_alpha_error < full_ldr_rgb_error)
		{
			best_error[i][2] = rgbs_alpha_error;
			format_of_choice[i][2] = FMT_RGB_SCALE_ALPHA;
		}
		else
		{
			best_error[i][2] = full_ldr_rgb_error;
			format_of_choice[i][2] = FMT_RGB;
		}

		// 4 integers can encode as RGBS or LA+LA
		float ldr_rgbs_error = quant_error_rgb + eci.alpha_drop_error + eci.rgb_scale_error;

		float lum_alpha_error = quant_error_rgba + eci.luminance_error;

		if (ldr_rgbs_error < lum_alpha_error)
		{
			best_error[i][1] = ldr_rgbs_error;
			format_of_choice[i][1] = FMT_RGB_SCALE;
		}
		else
		{
			best_error[i][1] = lum_alpha_error;
			format_of_choice[i][1] = FMT_LUMINANCE_ALPHA;
		}

		// 2 integers can encode as L+L
		float luminance_error = quant_error_rgb + eci.alpha_drop_error + eci.luminance_error;

		best_error[i][0] = luminance_error;
		format_of_choice[i][0] = FMT_LUMINANCE;
	}

#if ASTC_DEBUG_COUT
	for (int i = QUANT_6; i <= QUANT_256; i++)
	{
		for (int idx = 0; idx < 4; idx++)
		{
			std::cout << "endpoint format: " << quant_metod_str[i] << "\t" << end_point_format_str[format_of_choice[i][idx]] << "\tinteger count" << idx * 2 + 2 << "\terror:" << float(best_error[i][idx]) << std::endl;
		}

		int min_idx = 0;
		float min_error = 1e30;
		for (int idx = min_idx; idx < 4; idx++)
		{
			if (best_error[i][idx] < min_error)
			{
				min_error = best_error[i][idx];
				min_idx = idx;
			}
		}

		std::cout << "endpoint format: " << quant_metod_str[i] << "\t" << "best format:" << end_point_format_str[format_of_choice[i][min_idx]] << std::endl << std::endl;
	}

#endif
}

__device__ float one_partition_find_best_combination_for_bitcount(
	const float best_combined_error[21][4],
	const uint8_t best_combined_format[21][4],
	int bits_available,
	uint8_t& best_quant_level,
	uint8_t& best_format
)
{
	int best_integer_count = 0;
	float best_integer_count_error = ERROR_CALC_DEFAULT;
	for (int integer_count = 1; integer_count <= 4; integer_count++)
	{
		// Compute the quantization level for a given number of integers and a given number of bits
		int quant_level = quant_mode_table[integer_count][bits_available];

		// Don't have enough bits to represent a given endpoint format at all!
		if (quant_level < QUANT_6)
		{
			continue;
		}

		float integer_count_error = best_combined_error[quant_level][integer_count - 1];
		if (integer_count_error < best_integer_count_error)
		{
			best_integer_count_error = integer_count_error;
			best_integer_count = integer_count - 1;
		}
	}

	int ql = quant_mode_table[best_integer_count + 1][bits_available];

	best_quant_level = static_cast<uint8_t>(ql);
	best_format = FMT_LUMINANCE;

	if (ql >= QUANT_6)
	{
		best_format = best_combined_format[ql][best_integer_count];
	}

	return best_integer_count_error;
}


__device__ unsigned int compute_ideal_endpoint_formats(
	const image_block& blk,
	const endpoints& ep,
	// bitcounts and errors computed for the various quantization methods
	const int8_t* qwt_bitcounts,
	const float* qwt_errors,
	unsigned int start_block_mode,
	unsigned int end_block_mode,
	uint8_t best_ep_format_specifiers[TUNE_MAX_TRIAL_CANDIDATES],
	int block_mode[TUNE_MAX_TRIAL_CANDIDATES],
	quant_method quant_level[TUNE_MAX_TRIAL_CANDIDATES],
	compression_working_buffers& tmpbuf)
{
	// Compute the errors that result from various encoding choices (such as using luminance instead
	// of RGB, discarding Alpha, using RGB-scale in place of two separate RGB endpoints and so on)
	encoding_choice_errors eci;
	compute_encoding_choice_errors(blk, ep, eci);

	// 21 quant level, 4 integer count
	float best_error[21][4];
	uint8_t format_of_choice[21][4];
	compute_color_error_for_every_integer_count_and_quant_level(blk, eci, ep, best_error, format_of_choice);

	float* errors_of_best_combination = tmpbuf.errors_of_best_combination;

	uint8_t* best_quant_levels = tmpbuf.best_quant_levels;
	//uint8_t* best_quant_levels_mode = tmpbuf.best_quant_levels_mode;
	uint8_t* best_ep_formats = tmpbuf.best_ep_formats;

	float clear_error(ERROR_CALC_DEFAULT);
	int clear_quant(0);

	// Ensure that the first iteration understep contains data that will never be picked
	errors_of_best_combination[start_block_mode] = clear_error;
	best_quant_levels[start_block_mode] = clear_quant;

	// Ensure that last iteration overstep contains data that will never be picked
	errors_of_best_combination[end_block_mode - 1] = clear_error;
	best_quant_levels[end_block_mode - 1] = clear_quant;

	// Track a scalar best to avoid expensive search at least once ...
	float error_of_best_combination = ERROR_CALC_DEFAULT;
	int index_of_best_combination = -1;

	for (unsigned int i = start_block_mode; i < end_block_mode; i++)
	{
		if (qwt_errors[i] >= ERROR_CALC_DEFAULT)
		{
			errors_of_best_combination[i] = ERROR_CALC_DEFAULT;
			continue;
		}

		float error_of_best = one_partition_find_best_combination_for_bitcount(best_error, format_of_choice, qwt_bitcounts[i],
			best_quant_levels[i], best_ep_formats[i]);

#if ASTC_DEBUG_COUT
		std::cout << "Quant Mode Encoded: " <<i << "\t" << " best ep quant level:" << quant_metod_str[best_quant_levels[i]] << "\t" <<" best ep quant format:" <<"\t"<< end_point_format_str[best_ep_formats[i]]<<"\t" << " error:" << error_of_best << std::endl;
#endif

		float total_error = error_of_best + qwt_errors[i];
		errors_of_best_combination[i] = total_error;
		//best_quant_levels_mode[i] = best_quant_levels[i];

		if (total_error < error_of_best_combination)
		{
			error_of_best_combination = total_error;
			index_of_best_combination = i;
		}
	}

	int best_error_weights[TUNE_MAX_TRIAL_CANDIDATES];

	// Fast path the first result and avoid the list search for trial 0
	best_error_weights[0] = index_of_best_combination;
	if (index_of_best_combination >= 0)
	{
		errors_of_best_combination[index_of_best_combination] = ERROR_CALC_DEFAULT;
	}

	// Search the remaining results and pick the best candidate modes for trial 1+
	for (unsigned int i = 1; i < TUNE_MAX_TRIAL_CANDIDATES; i++)
	{
		int best_error_index(-1);
		float best_ep_error(ERROR_CALC_DEFAULT);

		for (unsigned int j = start_block_mode; j < end_block_mode; j++)
		{
			float err = errors_of_best_combination[j];
			bool is_better = err < best_ep_error;
			best_ep_error = is_better ? err : best_ep_error;
			best_error_index = is_better ? j : best_error_index;
		}

		best_error_weights[i] = best_error_index;

		// Max the error for this candidate so we don't pick it again
		if (best_error_index >= 0)
		{
			errors_of_best_combination[best_error_index] = ERROR_CALC_DEFAULT;
		}
		// Early-out if no more candidates are valid
		else
		{
			break;
		}
	}

	for (unsigned int i = 0; i < TUNE_MAX_TRIAL_CANDIDATES; i++)
	{
		if (best_error_weights[i] < 0)
		{
			return i;
		}

		block_mode[i] = best_error_weights[i];
		quant_level[i] = static_cast<quant_method>(best_quant_levels[best_error_weights[i]]);
		best_ep_format_specifiers[i] = best_ep_formats[best_error_weights[i]];
	}

	return TUNE_MAX_TRIAL_CANDIDATES;
}

#endif