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

// find best k-iteration
__shared__ uint32_t candidate_ep_format_specifiers[4+2];
__shared__ int candidate_block_mode_index[4+2];
__shared__ int candidate_color_quant_level[4+2];

__shared__ float candidate_combine_errors[4+2];

// recompute_ideal_colors_1plane
__shared__ float3 shared_data_mean;
__shared__ float3 shared_scale_dir;
__shared__ float4 shared_rgbs_color;

__shared__ float shared_block_compress_error;

struct endpoints
{
	float4 endpt0;
	float4 endpt1;
};

struct endpoints_and_weights
{
	endpoints ep;
	float weights[BLOCK_MAX_TEXELS];
};

struct symbolic_compressed_block
{
	endpoints_and_weights ei1;

	uint8_t block_type;
	uint16_t block_mode;
	quant_method quant_mode; // color quant mode
	float errorval;

	uint8_t color_formats;
	uint8_t weights[BLOCK_MAX_WEIGHTS];

	uint8_t color_values[8]; // * 
	int constant_color[4];
};

__shared__ symbolic_compressed_block shared_best_symbolic_block;


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

__device__ block_mode get_block_mode(const block_size_descriptor* const bsd, unsigned int block_mode)
{
	unsigned int packed_index = bsd->block_mode_packed_index[block_mode];
	return bsd->block_modes[packed_index];
}
#endif
