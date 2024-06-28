#ifndef _FGAC_DEVICE_HOST_COMMON_H_
#define _FGAC_DEVICE_HOST_COMMON_H_
#include <stdint.h>
#include <vector_types.h>

#define BLOCK_MAX_TEXELS (4*4)
#define WEIGHTS_MAX_BLOCK_MODES 2048
#define BLOCK_MAX_WEIGHTS 64
#define ERROR_CALC_DEFAULT 1e30f
#define TUNE_MAX_TRIAL_CANDIDATES 4
#define SYM_BTYPE_NONCONST 3
#define SYM_BTYPE_ERROR 0
#define SYM_BTYPE_CONST_U16 2
#define BLOCK_BAD_BLOCK_MODE 0xFFFFu
#define BLOCK_MIN_WEIGHT_BITS 24
#define BLOCK_MAX_WEIGHT_BITS 96
#define ASTC_MAGIC_ID 0x5CA1AB13;

#define CUDA_OUTBUFFER_DEBUG 0

struct astc_header
{
	uint8_t magic[4]; //ASTC_MAGIC_ID
	uint8_t block_x;
	uint8_t block_y;
	uint8_t block_z;
	uint8_t dim_x[3];			// dims = dim[0] + (dim[1] << 8) + (dim[2] << 16)
	uint8_t dim_y[3];			// Sizes are given in texels;
	uint8_t dim_z[3];			// block count is inferred
};

struct block_mode
{
	uint16_t mode_index;
	uint8_t quant_mode;
	uint8_t weight_bits;
};

enum quant_method
{
	QUANT_2 = 0,
	QUANT_3 = 1,
	QUANT_4 = 2,
	QUANT_5 = 3,
	QUANT_6 = 4,
	QUANT_8 = 5,
	QUANT_10 = 6,
	QUANT_12 = 7,
	QUANT_16 = 8,
	QUANT_20 = 9,
	QUANT_24 = 10,
	QUANT_32 = 11,

	QUANT_40 = 12,
	QUANT_48 = 13,
	QUANT_64 = 14,
	QUANT_80 = 15,
	QUANT_96 = 16,
	QUANT_128 = 17,
	QUANT_160 = 18,
	QUANT_192 = 19,
	QUANT_256 = 20
};

enum endpoint_formats
{
	FMT_LUMINANCE = 0,
	FMT_LUMINANCE_DELTA = 1,
	FMT_HDR_LUMINANCE_LARGE_RANGE = 2,
	FMT_HDR_LUMINANCE_SMALL_RANGE = 3,
	FMT_LUMINANCE_ALPHA = 4,
	FMT_LUMINANCE_ALPHA_DELTA = 5,
	FMT_RGB_SCALE = 6,
	FMT_HDR_RGB_SCALE = 7,
	FMT_RGB = 8,
	FMT_RGB_DELTA = 9,
	FMT_RGB_SCALE_ALPHA = 10,
	FMT_HDR_RGB = 11,
	FMT_RGBA = 12,
	FMT_RGBA_DELTA = 13,
	FMT_HDR_RGB_LDR_ALPHA = 14,
	FMT_HDR_RGBA = 15
};

struct block_size_descriptor
{
	uint8_t xdim; // The block x dimension
	uint8_t ydim; // The block y dimension
	uint8_t texel_count; // The block total texel count e.g. 8*8
	uint8_t padding;

	uint32_t block_mode_count_1plane_selected;

	uint16_t block_mode_packed_index[WEIGHTS_MAX_BLOCK_MODES];
	block_mode block_modes[WEIGHTS_MAX_BLOCK_MODES];
};
#endif
