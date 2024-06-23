#ifndef _FGAC_DEVICE_COMMON_H_
#define _FGAC_DEVICE_COMMON_H_
#include "../fgac_device_host_common.h"
#include "../common/helper_math.h"

struct quant_and_transfer_table
{
	uint8_t quant_to_unquant[32];
	uint8_t scramble_map[32];
};

// Ldr lumainance direct
// Ldr lumainance base + offset

// Ldr lumainance + alpha direct
// Ldr lumainance + alpha base + offset

// Ldr rgb base + scale
// Ldr rgb base + offset

// Ldr rgba base + offset

struct encoding_choice_errors
{
	float rgb_scale_error;
	float luminance_error;
	float alpha_drop_error;
};



struct line3
{
	float4 a;
	float4 b;
};

struct line4
{
	float4 a;
	float4 b;
};

struct processed_line3
{
	float4 amod;
	float4 bs;
};

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
	quant_method quant_mode;
	float errorval;
	
	uint8_t color_formats;
	uint8_t weights[BLOCK_MAX_WEIGHTS];

	uint8_t color_values[8];
	int constant_color[4];
};

struct compression_working_buffers
{
	endpoints_and_weights ei1;

	int8_t qwt_bitcounts[WEIGHTS_MAX_BLOCK_MODES];
	float qwt_errors[WEIGHTS_MAX_BLOCK_MODES];
	float errors_of_best_combination[WEIGHTS_MAX_BLOCK_MODES];
	
	uint8_t best_quant_levels[WEIGHTS_MAX_BLOCK_MODES];
	uint8_t best_ep_formats[WEIGHTS_MAX_BLOCK_MODES];

};

struct image_block
{
	float data_r[BLOCK_MAX_TEXELS];
	float data_g[BLOCK_MAX_TEXELS];
	float data_b[BLOCK_MAX_TEXELS];
	float data_a[BLOCK_MAX_TEXELS];

	uint8_t texel_count;

	float4 data_min;
	float4 data_max;
	float4 data_mean;
};

__device__ float4 normalize_safe(float4 a)
{
	float length = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
	if (length != 0.0f)
	{
		float inv_sqr_length = 1.0 / sqrt(length);
		return make_float4(a.x * inv_sqr_length, a.y * inv_sqr_length, a.z * inv_sqr_length, a.w * inv_sqr_length);
	}

	float val = 0.577350258827209473f;
	return make_float4(val, val, val, 0.0f);
}

__constant__ int quant_levels_map[21] =
{
	2,3,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96,128,160,192,256
};

__device__ unsigned int get_quant_level(quant_method method)
{
	return quant_levels_map[method];
}

__device__ block_mode get_block_mode(const block_size_descriptor& bsd, unsigned int block_mode)
{
	unsigned int packed_index = bsd.block_mode_packed_index[block_mode];
	return bsd.block_modes[packed_index];
}
#endif