#include "common.hlsl"

// {bitSplitPos, isTritsMode, isQuintsMode}
static const uint bitsTritsQuintsTable[QUANT_MAX * 3] =
{
	1, 0, 0,  // RANGE_2
	0, 1, 0,  // RANGE_3
	2, 0, 0,  // RANGE_4
	0, 0, 1,  // RANGE_5
	1, 1, 0,  // RANGE_6
	3, 0, 0,  // RANGE_8
	1, 0, 1,  // RANGE_10
	2, 1, 0,  // RANGE_12
	4, 0, 0,  // RANGE_16
	2, 0, 1,  // RANGE_20
	3, 1, 0,  // RANGE_24
	5, 0, 0,  // RANGE_32
	3, 0, 1,  // RANGE_40
	4, 1, 0,  // RANGE_48
	6, 0, 0,  // RANGE_64
	4, 0, 1,  // RANGE_80
	5, 1, 0,  // RANGE_96
	7, 0, 0,  // RANGE_128
	5, 0, 1,  // RANGE_160
	6, 1, 0,  // RANGE_192
	8, 0, 0   // RANGE_256
};

static const uint integerFromQuints[125] =
{
	0,1,2,3,4, 			8,9,10,11,12, 			16,17,18,19,20,			24,25,26,27,28, 		5,13,21,29,6,
	32,33,34,35,36, 	40,41,42,43,44, 		48,49,50,51,52, 		56,57,58,59,60, 		37,45,53,61,14,	
	64,65,66,67,68, 	72,73,74,75,76, 		80,81,82,83,84, 		88,89,90,91,92, 		69,77,85,93,22,
	96,97,98,99,100, 	104,105,106,107,108,	112,113,114,115,116,	120,121,122,123,124, 	101,109,117,125,30,	
	102,103,70,71,38, 	110,111,78,79,46, 		118,119,86,87,54, 		126,127,94,95,62, 		39,47,55,63,31
};

// RANGE_256 EpColor : bitoffset 0, 8 input number float4 * 2, bit count 8
void PackNumToBitPos(inout uint4 outPuts, inout uint bitoffset, uint number, uint bitCount)
{
	uint newBitPos = bitoffset + bitCount;

	uint nidx = newBitPos >> 5; // bitoffset / 32
	uint uidx = bitoffset >> 5; // newBitPos / 32
	uint byteOffset = bitoffset & 31;  // bitoffset % 32

	uint bytes[4] = {outPuts.x, outPuts.y, outPuts.z, outPuts.w};
	bytes[uidx] |= number << byteOffset;
	bytes[uidx + 1] |= (nidx > uidx) ? (number >> (32 - byteOffset)) : 0;

	outPuts.x = bytes[0];
	outPuts.y = bytes[1];
	outPuts.z = bytes[2];
	outPuts.w = bytes[3];

	bitoffset = newBitPos;
}

void SplitHighLow(uint splitNumber, uint splitPos, out uint high, out uint low)
{
	uint lowMask = uint(1 << splitPos) - 1;
	low = splitNumber & lowMask;
	high = (splitNumber >> splitPos) & 0xFF;
}

void EncodeTrits(uint bitcount, uint b0, uint b1, uint b2, uint b3, uint b4, inout uint4 outputs, inout uint outpos)
{

}

// example: RANGE_80
// RANGE_80: 4 0 1,  bitCount == 4,  output: 
// packhigh[5:6] low2[0:3] packhigh[3:4] low1[0:3] packhigh[0:2] low0[0:3]
// bitCount * 3 + 7 = 4 * 3 + 7 = 19
void EncodeQuints(uint bitCount, uint b0, uint b1, uint b2, inout uint4 outPuts, inout uint outPos)
{
	uint high0, high1, high2;
	uint low0, low1, low2;

	SplitHighLow(b0, bitCount, high0, low0);
	SplitHighLow(b1, bitCount, high1, low1);
	SplitHighLow(b2, bitCount, high2, low2);

	uint packhigh = integerFromQuints[high2 * 25 + high1 * 5 + high0];

	PackNumToBitPos(outPuts, outPos, low0, bitCount);	
	PackNumToBitPos(outPuts, outPos, packhigh & 7, 3);

	PackNumToBitPos(outPuts, outPos, low1, bitCount);
	PackNumToBitPos(outPuts, outPos, (packhigh >> 3) & 3, 3);

	PackNumToBitPos(outPuts, outPos, low2, bitCount);
	PackNumToBitPos(outPuts, outPos, (packhigh >> 5) & 3, 3);
}

void BiseEndpoints(uint numbers[8], int range/* default range is quant 256*/, inout uint4 outputs)
{
	const uint bitsSplitPos = bitsTritsQuintsTable[range * 3 + 0];
	const uint isTritsMode = bitsTritsQuintsTable[range * 3 + 1];
	const uint isQuintsMode = bitsTritsQuintsTable[range * 3 + 2];

#if HAS_ALPHA
	const int count = 8;
#else
	const int count = 6;
#endif

    if(isTritsMode == 1)
    {

    }
    else if(isQuintsMode == 1)
    {
		uint bitpos = 0;
		EncodeQuints(bitsSplitPos,numbers[0],numbers[1],numbers[2],outputs,bitpos);
		EncodeQuints(bitsSplitPos,numbers[3],numbers[4],numbers[5],outputs,bitpos);
		EncodeQuints(bitsSplitPos,numbers[6],numbers[7],0		  ,outputs,bitpos);
    }
    else
    {
        //todo
		uint bitpos = 0;
        for (int i = 0; i < count; ++i)
        {
			PackNumToBitPos(outputs, bitpos, numbers[i], bitsSplitPos);
        }
    }
}

// RANGE_6 1 1 0
void BiseWeights(uint numbers[16], int range, inout uint4 outputs)
{
	const uint bitsSplitPos = bitsTritsQuintsTable[range * 3 + 0];
	const uint isTritsMode = bitsTritsQuintsTable[range * 3 + 1];
	const uint isQuintsMode = bitsTritsQuintsTable[range * 3 + 2];

	if(isTritsMode == 1)
    {
		uint bitpos = 0;
		EncodeTrits(bitsSplitPos,numbers[0],numbers[1],numbers[2],numbers[3],numbers[4],outputs,bitpos);
		EncodeTrits(bitsSplitPos,numbers[5],numbers[6],numbers[7],numbers[8],numbers[9],outputs,bitpos);
		EncodeTrits(bitsSplitPos,numbers[10],numbers[11],numbers[12],numbers[13],numbers[14],outputs,bitpos);
		EncodeTrits(bitsSplitPos,numbers[15],0,0,0,0,outputs,bitpos);
    }
    else if(isQuintsMode == 1)
    {
		uint bitpos = 0;
		EncodeQuints(bitsSplitPos,numbers[0],numbers[1],numbers[2],outputs,bitpos);
		EncodeQuints(bitsSplitPos,numbers[3],numbers[4],numbers[5],outputs,bitpos);
		EncodeQuints(bitsSplitPos,numbers[6],numbers[7],numbers[8],outputs,bitpos);
		EncodeQuints(bitsSplitPos,numbers[6],numbers[7],numbers[8],outputs,bitpos);
		EncodeQuints(bitsSplitPos,numbers[9],numbers[10],numbers[11],outputs,bitpos);
		EncodeQuints(bitsSplitPos,numbers[12],numbers[13],numbers[14],outputs,bitpos);
		EncodeQuints(bitsSplitPos,numbers[15],0,0,outputs,bitpos);
    }
    else
    {
        //todo
		uint bitpos = 0;
        for (int i = 0; i < 16; ++i)
        {
			PackNumToBitPos(outputs, bitpos, numbers[i], bitsSplitPos);
        }
    }
}