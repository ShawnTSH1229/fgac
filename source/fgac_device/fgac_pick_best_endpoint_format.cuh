#ifndef _FGAC_PICK_BEST_ENDPOINT_FORMAT_CUH_
#define _FGAC_PICK_BEST_ENDPOINT_FORMAT_CUH_

#if CUDA_OUTBUFFER_DEBUG
#include <stdio.h>
#endif

#include "fgac_device_common.cuh"

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

__inline__ __device__ void compute_color_error_for_every_integer_count_and_quant_level(const block_size_descriptor* const bsd, uint32_t tid)
{
	int choice_error_idx = tid;
	if (choice_error_idx >= QUANT_2 && choice_error_idx < QUANT_6)
	{
		shared_best_error[choice_error_idx][3] = ERROR_CALC_DEFAULT;
		shared_best_error[choice_error_idx][2] = ERROR_CALC_DEFAULT;
		shared_best_error[choice_error_idx][1] = ERROR_CALC_DEFAULT;
		shared_best_error[choice_error_idx][0] = ERROR_CALC_DEFAULT;

		shared_format_of_choice[choice_error_idx][3] = FMT_RGBA;
		shared_format_of_choice[choice_error_idx][2] = FMT_RGB;
		shared_format_of_choice[choice_error_idx][1] = FMT_RGB_SCALE;
		shared_format_of_choice[choice_error_idx][0] = FMT_LUMINANCE;
	}

	float base_quant_error_rgb = 3 * bsd->texel_count;
	float base_quant_error_a = 1 * bsd->texel_count;
	float base_quant_error_rgba = base_quant_error_rgb + base_quant_error_a;

	if (choice_error_idx >= QUANT_6 && choice_error_idx <= QUANT_256)
	{
		float base_quant_error = baseline_quant_error[choice_error_idx - QUANT_6];
		float quant_error_rgb = base_quant_error_rgb * base_quant_error;
		float quant_error_rgba = base_quant_error_rgba * base_quant_error;

		// 8 integers can encode as RGBA+RGBA
		float full_ldr_rgba_error = quant_error_rgba;
		shared_best_error[choice_error_idx][3] = full_ldr_rgba_error;
		shared_format_of_choice[choice_error_idx][3] = FMT_RGBA;

		// 6 integers can encode as RGB+RGB or RGBS+AA
		float full_ldr_rgb_error = quant_error_rgb + 0;
		float rgbs_alpha_error = quant_error_rgba + shared_rgb_scale_error;

		if (rgbs_alpha_error < full_ldr_rgb_error)
		{
			shared_best_error[choice_error_idx][2] = rgbs_alpha_error;
			shared_format_of_choice[choice_error_idx][2] = FMT_RGB_SCALE_ALPHA;
		}
		else
		{
			shared_best_error[choice_error_idx][2] = full_ldr_rgb_error;
			shared_format_of_choice[choice_error_idx][2] = FMT_RGB;
		}

		// 4 integers can encode as RGBS or LA+LA
		float ldr_rgbs_error = quant_error_rgb + 0 + shared_rgb_scale_error;
		float lum_alpha_error = quant_error_rgba + shared_luminance_error;

		if (ldr_rgbs_error < lum_alpha_error)
		{
			shared_best_error[choice_error_idx][1] = ldr_rgbs_error;
			shared_format_of_choice[choice_error_idx][1] = FMT_RGB_SCALE;
		}
		else
		{
			shared_best_error[choice_error_idx][1] = lum_alpha_error;
			shared_format_of_choice[choice_error_idx][1] = FMT_LUMINANCE_ALPHA;
		}

		// 2 integers can encode as L+L
		float luminance_error = quant_error_rgb + 0 + shared_luminance_error;

		shared_best_error[choice_error_idx][0] = luminance_error;
		shared_format_of_choice[choice_error_idx][0] = FMT_LUMINANCE;

#if CUDA_OUTBUFFER_DEBUG
		printf("Lane Id %d: int count 2/4/6/8, error %f, error %f,  error %f,  error %f\n", choice_error_idx, shared_best_error[choice_error_idx][0], shared_best_error[choice_error_idx][1], shared_best_error[choice_error_idx][2], shared_best_error[choice_error_idx][3]);
#endif
	}
}

__inline__ __device__ void compute_encoding_choice_errors(const float3& data_mean,const float3& safe_dir,const float3& datav, const uint32_t& lane_id,const uint32_t& tid,const unsigned& mask)
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

	if (tid == 0)
	{
		shared_data_mean = data_mean;
		shared_scale_dir = samec_rgb_lines.b;
	}

	__syncwarp(mask);

	float sum_uncor_err = warp_reduce_sum(mask, uncor_err);
	float sum_samec_err = warp_reduce_sum(mask, samec_err);
	float sum_l_err = warp_reduce_sum(mask, l_err);

	if (lane_id == 0)
	{
		shared_rgb_scale_error = (sum_samec_err - sum_uncor_err) * 0.7f;// empirical
		shared_luminance_error = (sum_l_err - sum_uncor_err) * 3.0f;// empirical

#if CUDA_OUTBUFFER_DEBUG
		printf("RGB Scale Error: %f\n", (sum_samec_err - sum_uncor_err) * 0.7f);
		printf("RGB Lumin Error: %f\n", (sum_l_err - sum_uncor_err) * 3.0f);
#endif
	}
}

#endif