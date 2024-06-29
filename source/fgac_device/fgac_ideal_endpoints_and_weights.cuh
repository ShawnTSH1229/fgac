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

__inline__ __device__ void recompute_ideal_colors_1plane(const uint32_t& lane_id, const uint32_t& tid, const unsigned& mask)
{
	if (tid < BLOCK_MAX_TEXELS)
	{
		float scale = dot(float3(shared_scale_dir.x, shared_scale_dir.y, shared_scale_dir.z), float3(shared_blk_data[tid].x, shared_blk_data[tid].y, shared_blk_data[tid].z));
		float min_scale = scale;
		float max_scale = scale;
		__syncwarp(mask);
#pragma unroll
		for (int offset = (BLOCK_MAX_TEXELS >> 1); offset > 0; offset >>= 1)
		{
			float other_min_scale = __shfl_xor_sync(mask, min_scale, offset);
			if (min_scale > other_min_scale)
			{
				min_scale = other_min_scale;
			}

			float other_max_scale = __shfl_xor_sync(mask, max_scale, offset);
			if (max_scale < other_max_scale)
			{
				max_scale = other_max_scale;
			}
		}

		if (tid == 0)
		{
			float scalediv = min_scale / fmaxf(max_scale, 1e-10f);
			scalediv = clamp(scalediv, 0.0, 1.0);
			float3 sds = shared_scale_dir * max_scale;
			shared_rgbs_color = make_float4(sds.x, sds.y, sds.z, scalediv);
#if CUDA_OUTBUFFER_DEBUG
			printf("shared_rgbs_color %f,%f,%f,%f\n", shared_rgbs_color.x, shared_rgbs_color.y, shared_rgbs_color.z, shared_rgbs_color.w);
			printf("max_scale % f\n", max_scale);
			printf("min_scale % f\n", min_scale);
#endif
		}
	}
}

#endif