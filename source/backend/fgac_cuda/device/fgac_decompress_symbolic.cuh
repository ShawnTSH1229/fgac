#ifndef _FGAC_DECOMPRESS_SYMBOLIC_CUH_
#define _FGAC_DECOMPRESS_SYMBOLIC_CUH_
#include "fgac_decvice_common.cuh"


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
	else if(format == FMT_RGB_SCALE)
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


__device__ float compute_symbolic_block_difference_1plane_1partition(
	const endpoints_and_weights& ei,
	const block_size_descriptor& bsd,
	const symbolic_compressed_block& scb,
	const image_block& blk)
{
	// If we detected an error-block, blow up immediately.
	if (scb.block_type == SYM_BTYPE_ERROR)
	{
		return ERROR_CALC_DEFAULT;
	}


	int plane1_weights[BLOCK_MAX_TEXELS];
	for (unsigned int i = 0; i < bsd.texel_count; i++)
	{
		plane1_weights[i] = ei.weights[i] * 64;
	}

	// Decode the color endpoints for this partition
	int4 ep0;
	int4 ep1;
	bool rgb_lns;
	bool a_lns;

	unpack_color_endpoints(
		scb.color_formats,
		scb.color_values,
		ep0, ep1);

	float summa = 0;
	unsigned int texel_count = bsd.texel_count;
	for (unsigned int i = 0; i < texel_count; i++)
	{
		// Compute EP1 contribution
		int weight1 = plane1_weights[i];
		int ep1_r = ep1.x * weight1;
		int ep1_g = ep1.y * weight1;
		int ep1_b = ep1.z * weight1;
		int ep1_a = ep1.w * weight1;

		// Compute EP0 contribution
		int weight0 = int(64) - weight1;
		int ep0_r = ep0.x * weight0;
		int ep0_g = ep0.y * weight0;
		int ep0_b = ep0.z * weight0;
		int ep0_a = ep0.w * weight0;

		// Combine contributions
		int colori_r = int(ep0_r + ep1_r + int(32)) >> 6;
		int colori_g = int(ep0_g + ep1_g + int(32)) >> 6;
		int colori_b = int(ep0_b + ep1_b + int(32)) >> 6;
		int colori_a = int(ep0_a + ep1_a + int(32)) >> 6;

		// Compute color diff
		float color_r = float(colori_r);
		float color_g = float(colori_g);
		float color_b = float(colori_b);
		float color_a = float(colori_a);

		float color_orig_r = blk.data_r[i];
		float color_orig_g = blk.data_g[i];
		float color_orig_b = blk.data_b[i];
		float color_orig_a = blk.data_a[i];

		float color_error_r = fmin(abs(color_orig_r - color_r), float(1e15f));
		float color_error_g = fmin(abs(color_orig_g - color_g), float(1e15f));
		float color_error_b = fmin(abs(color_orig_b - color_b), float(1e15f));
		float color_error_a = fmin(abs(color_orig_a - color_a), float(1e15f));

		// Compute squared error metric
		color_error_r = color_error_r * color_error_r;
		color_error_g = color_error_g * color_error_g;
		color_error_b = color_error_b * color_error_b;
		color_error_a = color_error_a * color_error_a;

		float metric = color_error_r + color_error_g + color_error_b + color_error_a;

#if ASTC_DEBUG_COUT
		std::cout << "original color z: " << color_orig_b << " color_z: " << color_b << " metric:" << metric << std::endl;
		std::cout << "color_error_r: " << color_error_r << "color_error_g: " << color_error_g << " color_error_b:" << color_error_b << std::endl;
#endif
		// Mask off bad lanes
		summa += metric;
	}

	return summa;
}
#endif