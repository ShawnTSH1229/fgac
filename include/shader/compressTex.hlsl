// todo:
// 1. static const offset array 
// 2. find a better way to compute the end points
// 3. unroll
// 4. find the best block mode
// 5. swap function
// 6. dual plane
// 7. wave instruction optimization
// 8. minuint16 finalize block mode
// 9. support astc 8*8, current only support 6*6 and 4*4

#include "common.hlsl"
#include "ise.hlsl"
#include "weight_ise.hlsl"

uint sum(uint3 color)
{
	return color.r + color.g + color.b;
}

// todo: remove these functions
void swap(inout float4 lhs, inout float4 rhs)
{
	float4 tmp = lhs;
	lhs = rhs;
	rhs = tmp;
}

void swap(inout uint4 lhs, inout uint4 rhs)
{
	uint4 tmp = lhs;
	lhs = rhs;
	rhs = tmp;
}


SamplerState gSamPointWarp : register(s0, space1000);
SamplerState gSamLinearWarp : register(s4, space1000);
SamplerState gSamLinearClamp : register(s5, space1000);

Texture2D<float4> inputTexture             : register(t0);
RWStructuredBuffer<uint4> outputBuffer     : register(u0);

cbuffer fgacConstBuffer : register(b0)
{
    uint mipStartBlockIndex[MAX_MIP_NUM];
    uint m_groupNumX;
    uint m_blockNum;
    uint m_mipsNum;
    uint m_mip0TexWidth;
    uint m_mip0TexHeight;
    uint2 m_blockSize;
}

// TODO: static const offset array
float4 GetTexel(uint3 blockPixelStartPos, float2 mipInvSize, uint subBlockIndex)
{
    uint y = subBlockIndex / m_blockSize.x;
    uint x = subBlockIndex - y * m_blockSize.x;
    uint2 pixelPos = uint2(x, y) + blockPixelStartPos.xy;
    float2 pixelUV = pixelPos * mipInvSize;
    float4 texel = inputTexture.SampleLevel(gSamPointWarp, pixelUV, blockPixelStartPos.z);
#if IS_NORMALMAP
    texel.z = 1.0f;
    texel.w = 1.0f;
#endif
    return texel * 255.0f;
}

/*
* MaxAccumulationPixelDirection BEGIN
*/

void FindMinMaxFromBlock(const uint texelBlock[BLOCK_SIZE], float4 pixelMean, float4 maxDir, out float4 endPoint0, out float4 endPoint1)
{
    float minProjT =  1e30f;
    float maxProjT = -1e30f;

    for(uint index = 0; index < BLOCK_SIZE; index++)
    {
        float4 texel = UnpackTexel(texelBlock[index]);
        texel -= pixelMean;
        float projT = dot(texel,maxDir);
        minProjT = min(minProjT,projT);
        maxProjT = max(maxProjT,projT);
    }

    endPoint0 = clamp(maxDir * minProjT + pixelMean, 0.0f, 255.0f);
	endPoint1 = clamp(maxDir * maxProjT + pixelMean, 0.0f, 255.0f);

    //
    float4 e0u = round(endPoint0);
	float4 e1u = round(endPoint1);
	if (e0u.x + e0u.y + e0u.z > e1u.x + e1u.y + e1u.z)
	{
		swap(endPoint0, endPoint1);
	}
#if !HAS_ALPHA
	endPoint0.a = 255.0f;
	endPoint1.a = 255.0f;
#endif
}

// TODO: not a good solution, must modify this function
void MaxAccumulationPixelDirection(uint3 blockPixelStartPos, float2 mipInvSize, inout uint texelBlock[BLOCK_SIZE], out float4 endPoint0, out float4 endPoint1)
{
    float4 pixelMean = 0;

    //TODO
	for (uint index = 0; index < BLOCK_SIZE; index++)
    {
        float4 texel = GetTexel(blockPixelStartPos, mipInvSize, index);
        texelBlock[index] = PackTexel(texel);
        pixelMean += texel;
    }

    pixelMean /= BLOCK_SIZE;

    float4 sumr = 0;
    float4 sumg = 0;
    float4 sumb = 0;
    float4 suma = 0;

    for (uint texIndex = 0; texIndex < BLOCK_SIZE; texIndex++)
    {
        float4 texel = UnpackTexel(texelBlock[texIndex]);
        float4 dt = texel - pixelMean;
        sumr += (dt.x > 0) ? dt : 0;
		sumg += (dt.y > 0) ? dt : 0;
		sumb += (dt.z > 0) ? dt : 0;
		suma += (dt.w > 0) ? dt : 0;
    }

    float dotr = dot(sumr, sumr);
    float dotg = dot(sumg, sumg);
    float dotb = dot(sumb, sumb);
    float dota = dot(suma, suma);

    //todo: remove these dynamic branch
    float maxDot = dotr;
    float4 maxDir = sumr;

    if(dotg > maxDot)
    {
        maxDir = sumg;
        maxDot = dotg;
    }

    if(dotb > maxDot)
    {
        maxDir = sumb;
        maxDot = dotb;
    }

#if HAS_ALPHA
    if(dota > maxDot)
    {
        maxDir = suma;
        maxDot = dota;
    }
#endif

    float lenMaxDir = length(maxDir);
	maxDir = (lenMaxDir < SMALL_VALUE) ? maxDir : normalize(maxDir);

    FindMinMaxFromBlock(texelBlock, pixelMean, maxDir, endPoint0, endPoint1);
}

/*
* MaxAccumulationPixelDirection END
*/

/*
* EndPoint ISE BEGIN
*/
void EncodeColor(uint quantIndex/*unused*/, float4 e0, float4 e1, out uint endpointQuantized[8])
{
    uint4 e0q = round(e0);
	uint4 e1q = round(e1);
	endpointQuantized[0] = e0q.r;
	endpointQuantized[1] = e1q.r;
	endpointQuantized[2] = e0q.g;
	endpointQuantized[3] = e1q.g;
	endpointQuantized[4] = e0q.b;
	endpointQuantized[5] = e1q.b;
	endpointQuantized[6] = e0q.a;
	endpointQuantized[7] = e1q.a;
}

uint4 EndPointIse(uint colorQuantIndex /* use default quant method*/, float4 ep0, float4 ep1, uint endpointQuantmethod)
{
    uint epQuantized[8];
    EncodeColor(colorQuantIndex,ep0,ep1,epQuantized);
#if !HAS_ALPHA
	epQuantized[6] = 0;
	epQuantized[7] = 0;
#endif

    uint4 biseEp = 0;
    BiseEndpoints(epQuantized,endpointQuantmethod,biseEp);
    return biseEp;
}

/*
* EndPoint ISE END
*/


uint FinalizeBlockMode(uint weightQuantMethod)
{
    /*
    "Table C.2.8 - 2D Block Mode Layout".
	------------------------------------------------------------------------
	10  9   8   7   6   5   4   3   2   1   0   Width Height Notes
	------------------------------------------------------------------------
	D   H     B       A     R0  0   0   R2  R1  B + 4   A + 2
    */

	uint a = (Y_GRIDS - 2) & 0x3;
	uint b = (X_GRIDS - 4) & 0x3;

    uint dualPlane = 0;

    uint h = (weightQuantMethod < 6) ? 0 : 1;
    uint r = (weightQuantMethod % 6) + 2;

    uint blockMode = (r >> 1) & 0x3; // bit 0 and bit 1
    blockMode |= (r & 0x1) << 4; // bit 4
    blockMode |= (a & 0x3) << 5; // bit 5 and bit 6
    blockMode |= (b & 0x3) << 7; // bit 7 and bit 8
    blockMode |= h << 9;    // bit 9
    blockMode |= dualPlane << 10;   // bit 10

    return blockMode;
}
uint4 FinalizeBlock(uint blockmode, uint colorEndpointMode, uint partitionCount, uint partitionIndex, uint4 epIse, uint4 wtIse)
{
    uint4 finalizeBlock = uint4(0, 0, 0, 0);
    finalizeBlock.w = reversebits(wtIse.x); // weight bits [96:128]
    finalizeBlock.z = reversebits(wtIse.z); // weight bits [64:096] [81:096]
    finalizeBlock.y = reversebits(wtIse.y); 
    

    // block mode
    finalizeBlock.x = blockmode; // [0:10]

    // color end point mode
    finalizeBlock.x |= (colorEndpointMode & 0xF) << 13; // [11:14]

    // endpoints start from (multi_part ? bits 29 : bits 17)
    // assume partitionCount == 1
    finalizeBlock.x |= (epIse.x & 0x7FFF) << 17; // end points 15 bits [17:32]
	finalizeBlock.y = ((epIse.x >> 15) & 0x1FFFF); // end points 17 bits [32:49]
	finalizeBlock.y |= (epIse.y & 0x7FFF) << 17; // end points 15 bits [49:64]
	finalizeBlock.z |= ((epIse.y >> 15) & 0x1FFFF);// end points 12 bits [64:81]

    return finalizeBlock;
}

uint4 EncodeBlock(uint3 blockPixelStartPos, float2 mipInvSize)
{
    uint  texelBlock[BLOCK_SIZE];
    float4 endPoint0;
    float4 endPoint1;
    MaxAccumulationPixelDirection(blockPixelStartPos,mipInvSize,texelBlock,endPoint0,endPoint1);

    // TODO:
#if HAS_ALPHA
	uint4 bestBlockMode = uint4(QUANT_6, QUANT_256, 6, 7);
#else
	uint4 bestBlockMode = uint4(QUANT_12, QUANT_256, 12, 7);
#endif

    //todo: find the best quant method

    uint blockMode = FinalizeBlockMode(bestBlockMode.x);
    uint4 endPointIse = EndPointIse(bestBlockMode.w,endPoint0,endPoint1,bestBlockMode.y);
    uint4 weightIse = WeightIse(blockPixelStartPos,mipInvSize,texelBlock,bestBlockMode.z - 1,endPoint0,endPoint1,bestBlockMode.x);

#if HAS_ALPHA
	uint colorEndpointMode = CEM_LDR_RGBA_DIRECT;
#else
	uint colorEndpointMode = CEM_LDR_RGB_DIRECT;
#endif

    uint partitionCount = 1;
    uint partitionIndex = 0;
    return FinalizeBlock(blockMode, colorEndpointMode, partitionCount, partitionIndex, endPointIse, weightIse);
}   


[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void MainCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint rowBlockNum = m_groupNumX * GROUP_SIZE;
    uint blockIndex = dispatchThreadID.y * rowBlockNum + dispatchThreadID.x;

    if(blockIndex >= m_blockNum){return;}
    
    //TODO:
    uint mipLevel = 0;
    for(uint index = 1; index < m_mipsNum; index++) 
    {
        if(blockIndex < mipStartBlockIndex[index])
        {
            mipLevel = index - 1;
            break;
        }
    }

    uint2 curMipSize = uint2(m_mip0TexWidth >> mipLevel, m_mip0TexHeight >> mipLevel);
    uint2 curMipBlockNum = uint2(
        (curMipSize.x + m_blockSize.x - 1) / m_blockSize.x,
        (curMipSize.y + m_blockSize.y - 1) / m_blockSize.y);

    uint mipBlockIndex = blockIndex - mipStartBlockIndex[mipLevel];

    uint3 blockPixelStartPos;
    blockPixelStartPos.y = mipBlockIndex / curMipBlockNum.x;
    blockPixelStartPos.x = mipBlockIndex - blockPixelStartPos.y * curMipBlockNum.x;
    blockPixelStartPos.z = mipLevel;

    blockPixelStartPos.xy *= m_blockSize;

    float2 mipInvSize = float2(1.0f / curMipSize.x, 1.0f / curMipSize.y);
    outputBuffer[blockIndex] = EncodeBlock(blockPixelStartPos,mipInvSize);
}