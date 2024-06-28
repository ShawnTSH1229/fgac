#ifndef _FGAC_COLOR_QUANTIZE_CUH_
#define _FGAC_COLOR_QUANTIZE_CUH_

#include "fgac_quantization.cuh"
#include "fgac_decvice_common.cuh"

__inline__ __device__ uint8_t quant_color(
	quant_method quant_level,
	int value
) {
	int index = value * 2 + 1;
	return color_unquant_to_uquant_tables[quant_level - QUANT_6][index];
}

__inline__ __device__ uint8_t quant_color(
	quant_method quant_level,
	int value,
	float valuef
) {
	int index = value * 2;

	// Compute the residual to determine if we should round down or up ties.
	// Test should be residual >= 0, but empirical testing shows small bias helps.
	float residual = valuef - static_cast<float>(value);
	if (residual >= -0.1f)
	{
		index++;
	}

	return color_unquant_to_uquant_tables[quant_level - QUANT_6][index];
}

__inline__ __device__ int4 quant_color3(
	quant_method quant_level,
	int4 value,
	float4 valuef
)
{
	int4 index = value * 2;

	// Compute the residual to determine if we should round down or up ties.
	// Test should be residual >= 0, but empirical testing shows small bias helps.
	float4 residual = valuef - make_float4(value.x, value.y, value.z, value.w);

	index = int4(
		residual.x >= 0.1f ? index.x + 1 : index.x,
		residual.y >= 0.1f ? index.y + 1 : index.y,
		residual.z >= 0.1f ? index.z + 1 : index.z,
		residual.w >= 0.1f ? index.w + 1 : index.w
	);

	return int4(
		color_unquant_to_uquant_tables[quant_level - QUANT_6][index.x],
		color_unquant_to_uquant_tables[quant_level - QUANT_6][index.y],
		color_unquant_to_uquant_tables[quant_level - QUANT_6][index.z],
		0);
}

__inline__ __device__ void rgba_unpack(
	int4 input0,
	int4 input1,
	int4& output0,
	int4& output1
) 
{
	if ((input0.x + input0.y + input0.z) > (input1.x + input1.y + input1.z))
	{
		int4 input_temp = input0;
		input0 = input1;
		input1 = input_temp;
	}

	output0 = input0;
	output1 = input1;
}

__inline__ __device__ void quantize_rgb(
	float4 color0,
	float4 color1,
	int4& color0_out,
	int4& color1_out,
	quant_method quant_level
) {
	int4 color0i, color1i;
	float4 nudge = make_float4(0.2f);

	do
	{
		float4 round_color0 = color0 + make_float4(0.5f);
		
		int4 color0q = max(int4(round_color0.x, round_color0.y, round_color0.z, round_color0.w), make_int4(0));
		color0i = quant_color3(quant_level, color0q, color0);
		color0 = color0 - nudge;
	
		float4 round_color1 = color1 + make_float4(0.5f);
		int4 color1q = min(int4(round_color1.x, round_color1.y, round_color1.z, round_color1.w), make_int4(255));
		color1i = quant_color3(quant_level, color1q, color1);
		color1 = color1 + nudge;
	} while ((color0i.x + color0i.y + color0i.z) > (color1i.x + color1i.y + color1i.z));

	color0_out = color0i;
	color1_out = color1i;
}

__inline__ __device__ void quantize_rgba(
	float4 color0,
	float4 color1,
	int4& color0_out,
	int4& color1_out,
	quant_method quant_level
) {
	quantize_rgb(color0, color1, color0_out, color1_out, quant_level);

	float a0 = color0.w;
	float a1 = color1.w;

	color0_out.w = quant_color(quant_level, int(a0 + 0.5f), a0);
	color1_out.w = quant_color(quant_level, int(a1 + 0.5f), a1);
}

__inline__ __device__ void quantize_rgbs(
	float4 color,
	uint8_t output[4],
	quant_method quant_level
) {
	float scale = 1.0f / 257.0f;

	float r = clamp(color.x * scale, 0.0, 255.0);
	float g = clamp(color.y * scale, 0.0, 255.0);
	float b = clamp(color.z * scale, 0.0, 255.0);

	int ri = quant_color(quant_level, int(r + 0.5), r);
	int gi = quant_color(quant_level, int(g + 0.5), g);
	int bi = quant_color(quant_level, int(b + 0.5), b);

	float oldcolorsum = (color.x + color.y + color.z) * scale;
	float newcolorsum = static_cast<float>(ri + gi + bi);

	float scalea = clamp(color.w * (oldcolorsum + 1e-10f) / (newcolorsum + 1e-10f),0.0,1.0);
	int scale_idx = int(scalea * 256.0f + 0.5);
	scale_idx = clamp(scale_idx, 0, 255);

	output[0] = static_cast<uint8_t>(ri);
	output[1] = static_cast<uint8_t>(gi);
	output[2] = static_cast<uint8_t>(bi);
	output[3] = quant_color(quant_level, scale_idx);
}

__inline__ __device__ void quantize_rgbs_alpha(
	float4 color0,
	float4 color1,
	float4 color,
	uint8_t output[6],
	quant_method quant_level
) {
	float a0 = color0.w;
	float a1 = color1.w;

	output[4] = quant_color(quant_level, int(a0 + 0.5f), a0);
	output[5] = quant_color(quant_level, int(a1 + 0.5f), a1);

	quantize_rgbs(color, output, quant_level);
}

__inline__ __device__ void quantize_luminance(
	float4 color0,
	float4 color1,
	uint8_t output[2],
	quant_method quant_level
) {
	float lum0 = (color0.x + color0.y + color0.z) * (1.0f / 3.0f);
	float lum1 = (color1.x + color1.y + color1.z) * (1.0f / 3.0f);

	if (lum0 > lum1)
	{
		float avg = (lum0 + lum1) * 0.5f;
		lum0 = avg;
		lum1 = avg;
	}

	output[0] = quant_color(quant_level, int(lum0 + 0.5), lum0);
	output[1] = quant_color(quant_level, int(lum1 + 0.5), lum1);
}

/**
 * @brief Quantize a LDR LA color.
 *
 * @param      color0        The input unquantized color0 endpoint.
 * @param      color1        The input unquantized color1 endpoint.
 * @param[out] output        The output endpoints, returned as (l0, l1, a0, a1).
 * @param      quant_level   The quantization level to use.
 */
__inline__ __device__ void quantize_luminance_alpha(
	float4 color0,
	float4 color1,
	uint8_t output[4],
	quant_method quant_level
) {
	float lum0 = (color0.x + color0.y + color0.z) * (1.0f / 3.0f);
	float lum1 = (color1.x + color1.y + color1.z) * (1.0f / 3.0f);

	float a0 = color0.w;
	float a1 = color1.w;

	output[0] = quant_color(quant_level, int(lum0 + 0.5), lum0);
	output[1] = quant_color(quant_level, int(lum1 + 0.5), lum1);
	output[2] = quant_color(quant_level, int(a0 + 0.5), a0);
	output[3] = quant_color(quant_level, int(a1 + 0.5), a1);
}

__inline__ __device__ float get_rgba_encoding_error(
	float4 uquant0,
	float4 uquant1,
	int4 quant0,
	int4 quant1
) {
	float4 error0 = uquant0 - make_float4(quant0.x, quant0.y, quant0.z, quant0.w);
	float4 error1 = uquant1 - make_float4(quant1.x, quant1.y, quant1.z, quant1.w);
	float4 diff_error = (error0 * error0 + error1 * error1);
	return diff_error.x + diff_error.y + diff_error.z + diff_error.w;
}

__inline__ __device__ uint8_t pack_color_endpoints(
	float4 color0,
	float4 color1,
	float4 rgbs_color,
	int format,
	uint8_t* output,
	quant_method quant_level)
{
	// Clamp colors to a valid LDR range
	// Note that HDR has a lower max, handled in the conversion functions
	color0 = clamp(color0,make_float4(0.0f), make_float4(65535.0f));
	color1 = clamp(color1, make_float4(0.0f), make_float4(65535.0f));

	// Pre-scale the LDR value we need to the 0-255 quantizable range
	float4 color0_ldr = color0 * (1.0f / 257.0f);
	float4 color1_ldr = color1 * (1.0f / 257.0f);

	uint8_t retval = 0;
	int4 color0_out, color1_out;
	int4 color0_out2, color1_out2;
	
	if(format == FMT_RGBA)
	{
		quantize_rgba(color0_ldr, color1_ldr, color0_out2, color1_out2, quant_level);

		retval = FMT_RGBA;
		color0_out = color0_out2;
		color1_out = color1_out2;

		output[0] = static_cast<uint8_t>(color0_out.x);
		output[1] = static_cast<uint8_t>(color1_out.x);
		output[2] = static_cast<uint8_t>(color0_out.y);
		output[3] = static_cast<uint8_t>(color1_out.y);
		output[4] = static_cast<uint8_t>(color0_out.z);
		output[5] = static_cast<uint8_t>(color1_out.z);
		output[6] = static_cast<uint8_t>(color0_out.w);
		output[7] = static_cast<uint8_t>(color1_out.w);
	}
	else if (format == FMT_RGB_SCALE)
	{
		quantize_rgbs(rgbs_color, output, quant_level);
		retval = FMT_RGB_SCALE;
	}
	else if (format == FMT_RGB_SCALE_ALPHA)
	{
		quantize_rgbs_alpha(color0_ldr, color1_ldr, rgbs_color, output, quant_level);
		retval = FMT_RGB_SCALE_ALPHA;
	}
	else if (format == FMT_LUMINANCE_ALPHA)
	{
		quantize_luminance(color0_ldr, color1_ldr, output, quant_level);
		retval = FMT_LUMINANCE;
	}
	else if (format == FMT_LUMINANCE)
	{
		quantize_luminance_alpha(color0_ldr, color1_ldr, output, quant_level);
		retval = FMT_LUMINANCE_ALPHA;
	}
	else
	{
		//format == FMT_RGB
		quantize_rgb(color0_ldr, color1_ldr, color0_out2, color1_out2, quant_level);

		retval = FMT_RGB;
		color0_out = color0_out2;
		color1_out = color1_out2;

		output[0] = uint8_t(color0_out.x);
		output[1] = uint8_t(color1_out.x);
		output[2] = uint8_t(color0_out.y);
		output[3] = uint8_t(color1_out.y);
		output[4] = uint8_t(color0_out.z);
		output[5] = uint8_t(color1_out.z);
		output[6] = uint8_t(255);
		output[7] = uint8_t(255);
	}
	return retval;
}

#endif