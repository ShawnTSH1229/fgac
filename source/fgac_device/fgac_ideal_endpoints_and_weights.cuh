#ifndef _FGAC_IDEAL_ENDPOINTS_AND_WEIGHTS_CUH_
#define _FGAC_IDEAL_ENDPOINTS_AND_WEIGHTS_CUH_

#include "fgac_device_common.cuh"

__inline__ __device__ void compute_ideal_colors_and_weights_4_comp(float3 datav, uint32_t lane_id, uint32_t tid, unsigned mask, float3 data_mean, float3& safe_dir)
{
	__syncwarp(mask);
	safe_dir = warp_boardcast_vec(mask, safe_dir);
	const float param = dot(datav - data_mean, safe_dir);
	float lowparam = param;
	float highparam = param;

	__syncwarp(mask);
#pragma unroll
	for (int offset = (BLOCK_MAX_TEXELS >> 1); offset > 0; offset >>= 1)
	{
		float other_low_param = __shfl_xor_sync(mask, lowparam, offset);
		if (lowparam > other_low_param)
		{
			lowparam = other_low_param;
		}

		float other_highparam = __shfl_xor_sync(mask, highparam, offset);
		if (highparam < other_highparam)
		{
			highparam = other_highparam;
		}
	}

	float scale;

	if (lane_id == 0)
	{
		if (highparam <= lowparam)
		{
			lowparam = 0.0f;
			highparam = 1e-7f;
		}

		float3 endpt0 = data_mean + safe_dir * lowparam;
		float3 endpt1 = data_mean + safe_dir * highparam;

		float length = highparam - lowparam;
		scale = 1.0f / length;
		shared_endpt[0] = endpt0;
		shared_endpt[1] = endpt1;
	}

	__syncwarp(mask);

	scale = __shfl_sync(mask, scale, 0, BLOCK_MAX_TEXELS);

	float idx = (param - lowparam) * scale;
	idx = clamp(idx, 0.0, 1.0);
	float weight = idx;

	shared_blk_weights[tid] = weight;
}

#endif