#include "common.hlsl"
#include "astc_table.hlsl"
#include "ise.hlsl"

uint QuantizeWeight(uint weightRange, float weight)
{
	uint q = round(weight * weightRange);
	return clamp(q, 0, weightRange);
}

void CalculateNormalWeights(uint3 blockPixelStartPos, float2 mipInvSize, const uint texelBlock[BLOCK_SIZE], float4 ep0, float4 ep1, out float projectedWeight[X_GRIDS * Y_GRIDS])
{
    int i = 0;
    float4 endPointDir = ep1 - ep0;
    if(length(endPointDir) < 1e-10)
    {
        //todo:
        for (i = 0; i < X_GRIDS * Y_GRIDS; ++i)
		{
			projectedWeight[i] = 0;
		}
    }
    else
    {
        endPointDir = normalize(endPointDir);
		float minw = 1e31f;
		float maxw = -1e31f;

        #if BLOCK_6X6

        #else
        for (i = 0; i < BLOCK_SIZE; ++i)
        {
            float4 texel = UnpackTexel(texelBlock[i]);

			float w = dot(endPointDir, texel - ep0);
			minw = min(w, minw);
			maxw = max(w, maxw);
			projectedWeight[i] = w;
        }
        #endif

        float invlen = maxw - minw;
		invlen = max(SMALL_VALUE, invlen);
		invlen = 1.0f / invlen;
		
		for (i = 0; i < X_GRIDS * Y_GRIDS; ++i)
		{
			projectedWeight[i] = (projectedWeight[i] - minw) * invlen;
		}
    }
}

void QuantizeWeights(float projectedWeight[X_GRIDS * Y_GRIDS], uint weightRange, out uint weights[X_GRIDS * Y_GRIDS])
{
    for(uint i = 0; i < X_GRIDS * Y_GRIDS; i++)
    {
        weights[i] = QuantizeWeight(weightRange,projectedWeight[i]);
    }
}

void CalculateQuantizedWeights(uint3 blockPixelStartPos, float2 mipInvSize, const uint texelBlock[BLOCK_SIZE], uint weightRange, float4 ep0, float4 ep1, out uint weights[X_GRIDS * Y_GRIDS])
{
    float projectedWeight[X_GRIDS * Y_GRIDS];
    CalculateNormalWeights(blockPixelStartPos,mipInvSize,texelBlock,ep0,ep1,projectedWeight);
    QuantizeWeights(projectedWeight, weightRange, weights);
}

//default weightRange 6 - 1 = 5, weightQuantmethod QUANT_6 = 4
uint4 WeightIse(uint3 blockPixelStartPos, float2 mipInvSize, const uint texelBlock[BLOCK_SIZE], uint weightRange, float4 ep0, float4 ep1, uint  weightQuantmethod)
{
    uint weightQuantized[X_GRIDS * Y_GRIDS];
    CalculateQuantizedWeights(blockPixelStartPos, mipInvSize, texelBlock, weightRange, ep0, ep1, weightQuantized);

    for (int i = 0; i < X_GRIDS * Y_GRIDS; ++i)
	{
		int w = weightQuantmethod * WEIGHT_QUANTIZE_NUM + weightQuantized[i];
		weightQuantized[i] = scrambleTable[w];
	}

    //scrambleTable
    uint4 weightIseOut = 0;
    BiseWeights(weightQuantized,weightQuantmethod,weightIseOut);
    return weightIseOut;
}