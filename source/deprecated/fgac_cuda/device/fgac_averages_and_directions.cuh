#ifndef _FGAC_AVERAGES_AND_DIRECTION_CUH_
#define _FGAC_AVERAGES_AND_DIRECTION_CUH_
#include "fgac_decvice_common.cuh"

__device__ void compute_avgs_and_dirs_3_comp(const image_block& blk, float4& dir)
{
	float4 sum_xp(0.0);
	float4 sum_yp(0.0);
	float4 sum_zp(0.0);

	for (unsigned int i = 0; i < blk.texel_count; i++)
	{
		float4 texel_datum = make_float4(blk.data_r[i], blk.data_g[i], blk.data_b[i], blk.data_a[i]);
		texel_datum = texel_datum - blk.data_mean;
		texel_datum.w = 0.0;

		sum_xp += (texel_datum.x > 0) ? texel_datum : make_float4(0);
		sum_yp += (texel_datum.y > 0) ? texel_datum : make_float4(0);
		sum_zp += (texel_datum.z > 0) ? texel_datum : make_float4(0);
	}

	float prod_xp = dot(sum_xp, sum_xp);
	float prod_yp = dot(sum_yp, sum_yp);
	float prod_zp = dot(sum_zp, sum_zp);

	float4 best_vector = sum_xp;
	float best_sum = prod_xp;

	if (prod_yp > best_sum)
	{
		best_vector = sum_yp;
		best_sum = prod_yp;
	}

	if (prod_zp > best_sum)
	{
		best_vector = sum_zp;
		best_sum = prod_zp;
	}

	dir = best_vector;
}

__device__ void compute_avgs_and_dirs_4_comp(const image_block& blk, float4& dir)
{
	float4 sum_xp(0.0);
	float4 sum_yp(0.0);
	float4 sum_zp(0.0);
	float4 sum_wp(0.0);

	for (unsigned int i = 0; i < blk.texel_count; i++)
	{
		float4 texel_datum = make_float4(blk.data_r[i], blk.data_g[i], blk.data_b[i], blk.data_a[i]);
		texel_datum = texel_datum - blk.data_mean;

		sum_xp += (texel_datum.x > 0) ? texel_datum : make_float4(0);
		sum_yp += (texel_datum.y > 0) ? texel_datum : make_float4(0);
		sum_zp += (texel_datum.z > 0) ? texel_datum : make_float4(0);
		sum_wp += (texel_datum.w > 0) ? texel_datum : make_float4(0);
	}

	float prod_xp = dot(sum_xp, sum_xp);
	float prod_yp = dot(sum_yp, sum_yp);
	float prod_zp = dot(sum_zp, sum_zp);
	float prod_wp = dot(sum_wp, sum_wp);

	float4 best_vector = sum_xp;
	float best_sum = prod_xp;

	if (prod_yp > best_sum)
	{
		best_vector = sum_yp;
		best_sum = prod_yp;
	}

	if (prod_zp > best_sum)
	{
		best_vector = sum_zp;
		best_sum = prod_zp;
	}

	if (prod_wp > best_sum)
	{
		best_vector = sum_wp;
		best_sum = prod_wp;
	}

	dir = best_vector;
}
#endif