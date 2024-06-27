#ifndef _FGAC_DECOMPRESS_SYMBOLIC_CUH_
#define _FGAC_DECOMPRESS_SYMBOLIC_CUH_
#include "fgac_device_common.cuh"

__inline__ __device__ void rgb_scale_unpack(
	int4 input0,
	int scale,
	int4& output0,
	int4& output1
) {
	output1 = input0;
	output1.w = 255;

	int4 scaled_value = input0 * scale;

	output0 = int4(scaled_value.x >> 8, scaled_value.y >> 8, scaled_value.z >> 8, scaled_value.w >> 8);
	output0.w = 255;
}

__inline__ __device__ void rgb_scale_alpha_unpack(
	int4 input0,
	uint8_t alpha1,
	uint8_t scale,
	int4& output0,
	int4& output1
) {
	output1 = input0;
	output1.w = alpha1;

	int4 scaled_value = input0 * scale;
	output0 = int4(scaled_value.x >> 8, scaled_value.y >> 8, scaled_value.z >> 8, scaled_value.w >> 8);
	output0.w = input0.w;
}

__inline__ __device__ void luminance_unpack(
	const uint8_t input[2],
	int4& output0,
	int4& output1
) {
	int lum0 = input[0];
	int lum1 = input[1];
	output0 = int4(lum0, lum0, lum0, 255);
	output1 = int4(lum1, lum1, lum1, 255);
}

__inline__ __device__ void luminance_alpha_unpack(
	const uint8_t input[4],
	int4& output0,
	int4& output1
) {
	int lum0 = input[0];
	int lum1 = input[1];
	int alpha0 = input[2];
	int alpha1 = input[3];
	output0 = int4(lum0, lum0, lum0, alpha0);
	output1 = int4(lum1, lum1, lum1, alpha1);
}


__inline__ __device__ void unpack_color_endpoints(
	int format,
	const uint8_t* input,
	int4& output0,
	int4& output1
) {

	if (format == FMT_RGB)
	{
		int4 input0q(input[0], input[2], input[4], 255);
		int4 input1q(input[1], input[3], input[5], 255);

		output0 = input0q;
		output1 = input1q;
	}
	else if (format == FMT_RGBA)
	{
		int4 input0q(input[0], input[2], input[4], input[6]);
		int4 input1q(input[1], input[3], input[5], input[7]);
		output0 = input0q;
		output1 = input1q;
	}
	else if (format == FMT_RGB_SCALE)
	{
		int4 input0q(input[0], input[1], input[2], 0);
		uint8_t scale = input[3];
		rgb_scale_unpack(input0q, scale, output0, output1);
	}
	else if (format == FMT_RGB_SCALE_ALPHA)
	{
		int4 input0q(input[0], input[1], input[2], input[4]);
		uint8_t alpha1q = input[5];
		uint8_t scaleq = input[3];
		rgb_scale_alpha_unpack(input0q, alpha1q, scaleq, output0, output1);
	}
	else if (format == FMT_LUMINANCE)
	{
		luminance_unpack(input, output0, output1);
	}
	else if (format == FMT_LUMINANCE_ALPHA)
	{
		luminance_alpha_unpack(input, output0, output1);
	}
	output0 = int4(output0.x * 257, output0.y * 257, output0.z * 257, output0.w * 257);
	output1 = int4(output1.x * 257, output1.y * 257, output1.z * 257, output1.w * 257);
}

#endif