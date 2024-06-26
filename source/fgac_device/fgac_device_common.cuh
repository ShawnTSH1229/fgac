#ifndef _FGAC_DEVICE_COMMON_H_
#define _FGAC_DEVICE_COMMON_H_
#include "../fgac_common/fgac_device_host_common.h"
#include "../fgac_common/helper_math.h"

__shared__ float shared_blk_weights[BLOCK_MAX_TEXELS];
__shared__ float3 shared_blk_data[BLOCK_MAX_TEXELS];
__shared__ float3 shared_endpt[2];
__shared__ float shared_rgb_scale_error;
__shared__ float shared_luminance_error;

// 21 quant level, 4 integer count
__shared__ float shared_best_error[21][4];
__shared__ uint8_t shared_format_of_choice[21][4];


__shared__ uint32_t candidate_ep_format_specifiers[4+2];
__shared__ int candidate_block_mode_index[4+2];
__shared__ int candidate_color_quant_level[4+2];

__shared__ float candidate_combine_errors[4+2];

struct quant_and_transfer_table
{
	uint8_t quant_to_unquant[32];
	uint8_t scramble_map[32];
};

struct line3
{
	float3 a;
	float3 b;
};

struct processed_line3
{
	float3 amod;
	float3 bs;
};

__inline__ __device__ float3 normalize_safe(float3 a)
{
	float length = a.x * a.x + a.y * a.y + a.z * a.z;
	if (length != 0.0f)
	{
		float inv_sqr_length = 1.0 / sqrt(length);
		return make_float3(a.x * inv_sqr_length, a.y * inv_sqr_length, a.z * inv_sqr_length);
	}

	float val = 0.577350258827209473f;
	return make_float3(val, val, val);
}

__constant__ int quant_levels_map[21] =
{
	2,3,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96,128,160,192,256
};

__inline__ __device__ unsigned int get_quant_level(quant_method method)
{
	return quant_levels_map[method];
}
#endif
