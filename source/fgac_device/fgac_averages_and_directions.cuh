#ifndef _FGAC_AVERAGES_AND_DIRECTION_CUH_
#define _FGAC_AVERAGES_AND_DIRECTION_CUH_
#include "fgac_device_common.cuh"
__inline__ __device__ float3 compute_avgs_and_dirs_3_comp(float3 datav,float3 data_mean, uint32_t lane_id, unsigned mask)
{
	float3 safe_dir;
	float3 texel_datum = datav - data_mean;

	float3 valid_sum_xp = (texel_datum.x != 0 ? texel_datum : float3(0, 0, 0));
	float3 valid_sum_yp = (texel_datum.x != 0 ? texel_datum : float3(0, 0, 0));
	float3 valid_sum_zp = (texel_datum.x != 0 ? texel_datum : float3(0, 0, 0));

	float3 sum_xp = warp_reduce_vec_sum(mask, valid_sum_xp);
	float3 sum_yp = warp_reduce_vec_sum(mask, valid_sum_yp);
	float3 sum_zp = warp_reduce_vec_sum(mask, valid_sum_zp);

	if (lane_id == 0)
	{
		float prod_xp = dot(sum_xp, sum_xp);
		float prod_yp = dot(sum_yp, sum_yp);
		float prod_zp = dot(sum_zp, sum_zp);

		float3 best_vector = sum_xp;
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

		if ((best_vector.x + best_vector.y + best_vector.z) < 0.0f)
		{
			best_vector = -best_vector;
		}

		float length_dir = length(best_vector);
		safe_dir = (length_dir < 1e-10) ? normalize(make_float3(1.0)) : normalize(best_vector);
	}
	return safe_dir;
}
#endif