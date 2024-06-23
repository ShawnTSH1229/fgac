#ifndef _FGAC_IDEAL_ENDPOINTS_AND_WEIGHTS_CUH_
#define _FGAC_IDEAL_ENDPOINTS_AND_WEIGHTS_CUH_
#include "fgac_decvice_common.cuh"
#include "fgac_averages_and_directions.cuh"
#include "fgac_quantization.cuh"
#include "fgac_weight_quant_xfer_tables.cuh"

#if ASTC_DEBUG_COUT
#include "stb_image_write.h"
#endif

__device__ void compute_ideal_colors_and_weights_4_comp(const image_block& blk, endpoints_and_weights& ei)
{
	float4 dir;
	compute_avgs_and_dirs_4_comp(blk, dir);

	if ((dir.x + dir.y + dir.z) < 0.0f)
	{
		dir = -dir;
	}

	float length_dir = length(dir);
	line4 line{ blk.data_mean, length_dir < 1e-10 ? normalize(make_float4(1.0)) : normalize(dir) };
	float lowparam{ 1e10f };
	float highparam{ -1e10f };

	for (unsigned int j = 0; j < blk.texel_count; j++)
	{
		float4 point(blk.data_r[j], blk.data_g[j], blk.data_b[j], blk.data_a[j]);
		float param = dot(point - line.a, line.b);

		ei.weights[j] = param;

		lowparam = fmin(param, lowparam);
		highparam = fmax(param, highparam);
	}

	if (highparam <= lowparam)
	{
		lowparam = 0.0f;
		highparam = 1e-7f;
	}

	ei.ep.endpt0 = line.a + line.b * lowparam;
	ei.ep.endpt1 = line.a + line.b * highparam;

	float length = highparam - lowparam;
	float scale = 1.0f / length;

	for (unsigned int j = 0; j < blk.texel_count; j++)
	{
		float idx = (ei.weights[j] - lowparam) * scale;
		idx = clamp(idx, 0.0, 1.0);
		ei.weights[j] = idx;
	}

#if ASTC_DEBUG_COUT

	std::cout << "Mean:" << line.a.x / 65536.0 << "," << line.a.y / 65536.0 << "," << line.a.z / 65536.0 << std::endl;
	std::cout << "Direction:" << line.b.x << "," << line.b.y << "," << line.b.z << std::endl;

	std::cout << "End Point0:" << ei.ep.endpt0.x / 65536.0 << "," << ei.ep.endpt0.y / 65536.0 << "," << ei.ep.endpt0.z / 65536.0 << std::endl;
	std::cout << "End Point1:" << ei.ep.endpt1.x / 65536.0 << "," << ei.ep.endpt1.y / 65536.0 << "," << ei.ep.endpt1.z / 65536.0 << std::endl;

	int w_idx = 0;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			std::cout << "W:" << (ei.weights[w_idx] / scale + lowparam) / 65536.0 << ", ";
			w_idx++;
		}
		std::cout << std::endl;
	}

	int write_idx = 0;
	char vis_data[4 * 4 * 4];

	int col_ifx = 0;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			float4 intered_color = (ei.ep.endpt0 + (ei.ep.endpt1 - ei.ep.endpt0) * ei.weights[col_ifx]) / 65536.0f;
			std::cout << "New Color:" << intered_color.x << ", " << intered_color.y << ", " << intered_color.z << " ";
			col_ifx++;

			vis_data[write_idx + 0] = char(int(intered_color.x * 255));
			vis_data[write_idx + 1] = char(int(intered_color.y * 255));
			vis_data[write_idx + 2] = char(int(intered_color.z * 255));
			vis_data[write_idx + 3] = char(255);
			write_idx += 4;
		}
		std::cout << std::endl;
	}

	stbi_write_tga("H:/ShawnTSH1229/fgac/tex_test_4_4.tga", 4, 4, 4, vis_data);
#endif
}

__device__ void compute_ideal_colors_and_weights_1plane(
	const image_block& blk,
	endpoints_and_weights& ei
) {
	compute_ideal_colors_and_weights_4_comp(blk, ei);
}

__device__ float compute_error_of_weight_set_1plane(
	const endpoints_and_weights& eai,
	const block_size_descriptor& bsd,
	const float* weight_quant_uvalue)
{
	float error_summa = 0;
	for (unsigned int i = 0; i < bsd.texel_count; i++)
	{
		// Load the weight set directly, without interpolation
		float current_values = weight_quant_uvalue[i];

		// Compute the error between the computed value and the ideal weight
		float actual_values = eai.weights[i];
		float diff = current_values - actual_values;
		float error = diff * diff;

		error_summa += error;
	}
	return error_summa;
}

// The available quant levels, stored with a minus 1 bias
__constant__ float quant_levels_m1[12]{
	1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f, 9.0f, 11.0f, 15.0f, 19.0f, 23.0f, 31.0f
};

__device__ void compute_quantized_weights_for_decimation(
	const block_size_descriptor& bsd,
	const endpoints_and_weights& ei,
	float* weight_set_out,
	quant_method quant_level
)
{
	int weight_count = bsd.texel_count;
	const quant_and_transfer_table& qat = quant_and_xfer_tables[quant_level];
	
	uint quant_level_map = get_quant_level(quant_level);
	int steps_m1(quant_level_map - 1);
	float quant_level_m1 = quant_levels_m1[quant_level];

	float rscale = 1.0f / 64.0f;

	for (int i = 0; i < weight_count; i++)
	{
		float ix = ei.weights[i];
		ix = clamp(ix, 0.0, 1.0);

		// Look up the two closest indexes and return the one that was closest(quant)
		float ix1 = ix * quant_level_m1;

		int weightl = int(ix1);
		int weighth = min(weightl + 1, steps_m1);

		float ixl = qat.quant_to_unquant[weightl];
		float ixh = qat.quant_to_unquant[weighth];

		if (ixl + ixh < 128.0f * ix)
		{
			ixl = ixh;
		}
		else
		{
			ixl = ixl;
		}

		// Invert the weight-scaling that was done initially
		weight_set_out[i] = ixl * rscale;
	}
}

__device__ void recompute_ideal_colors_1plane(
	const image_block& blk,
	float4& rgbs_vectors)
{
	float4 rgba_sum(blk.data_mean);
	float4 scale_dir = normalize(make_float4(rgba_sum.x, rgba_sum.y, rgba_sum.z, 0));
	
	float scale_max = 0.0f;
	float scale_min = 1e10f;

	for (unsigned int j = 0; j < blk.texel_count; j++)
	{
		float4 rgba = make_float4(blk.data_r[j], blk.data_g[j], blk.data_b[j], blk.data_a[j]);
		float scale = dot(float3(scale_dir.x, scale_dir.x, scale_dir.x), float3(rgba.x, rgba.y, rgba.z));
		scale_min = min(scale, scale_min);
		scale_max = max(scale, scale_max);
	}

	float scalediv = scale_min / max(scale_max, 1e-10f);
	scalediv = clamp(scalediv,0.0,1.0);


	float4 sds = scale_dir * scale_max;
	rgbs_vectors = make_float4(sds.x, sds.y, sds.z, scalediv);
}
#endif