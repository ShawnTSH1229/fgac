#define GROUP_SIZE 8
#define MAX_MIP_NUM 14
#define BLOCK_SIZE (4 * 4) //TODO:
#define IS_NORMALMAP 0
#define HAS_ALPHA 1
#define SMALL_VALUE 1e-6

#define BLOCK_6X6 0

#define X_GRIDS 4
#define Y_GRIDS 4

// arm astc-enmcoder astcenc_internal.h quant metod begin
#define	QUANT_2 0
#define	QUANT_3 1
#define	QUANT_4 2
#define	QUANT_5 3
#define	QUANT_6 4
#define	QUANT_8 5
#define	QUANT_10 6
#define	QUANT_12 7
#define	QUANT_16 8
#define	QUANT_20 9
#define	QUANT_24 10
#define	QUANT_32 11
#define	QUANT_40 12
#define	QUANT_48 13
#define	QUANT_64 14
#define	QUANT_80 15
#define	QUANT_96 16
#define	QUANT_128 17
#define	QUANT_160 18
#define	QUANT_192 19
#define	QUANT_256 20
#define	QUANT_MAX 21
// arm astc-enmcoder astcenc_internal.h quant metod endd

#define CEM_LDR_RGB_DIRECT 8
#define CEM_LDR_RGBA_DIRECT 12

uint PackTexel(float4 texel)
{
    uint4 packedTexel = (uint4)texel;
    return (packedTexel.x << 0) | (packedTexel.x << 8) | (packedTexel.x << 16) | (packedTexel.x << 24);
}

float4 UnpackTexel(uint packedTexel)
{
	uint4 unpackedTexel;
	unpackedTexel.x = (packedTexel >> 0) & 0xff;
	unpackedTexel.y = (packedTexel >> 8) & 0xff;
	unpackedTexel.z = (packedTexel >> 16) & 0xff;
	unpackedTexel.w = (packedTexel >> 24) & 0xff;
	return (float4)unpackedTexel;
}