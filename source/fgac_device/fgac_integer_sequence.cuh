#ifndef _FGAC_INT_SEQUENCE_CUH_
#define _FGAC_INT_SEQUENCE_CUH_

#include "fgac_device_common.cuh"

struct ise_size
{
	/** @brief The scaling parameter. */
	uint8_t scale : 6;

	/** @brief The divisor parameter. */
	uint8_t divisor : 2;
};

__constant__ ise_size  ise_sizes[21]
{
	{  1, 0 }, // QUANT_2
	{  8, 2 }, // QUANT_3
	{  2, 0 }, // QUANT_4
	{  7, 1 }, // QUANT_5
	{ 13, 2 }, // QUANT_6
	{  3, 0 }, // QUANT_8
	{ 10, 1 }, // QUANT_10
	{ 18, 2 }, // QUANT_12
	{  4, 0 }, // QUANT_16
	{ 13, 1 }, // QUANT_20
	{ 23, 2 }, // QUANT_24
	{  5, 0 }, // QUANT_32
	{ 16, 1 }, // QUANT_40
	{ 28, 2 }, // QUANT_48
	{  6, 0 }, // QUANT_64
	{ 19, 1 }, // QUANT_80
	{ 33, 2 }, // QUANT_96
	{  7, 0 }, // QUANT_128
	{ 22, 1 }, // QUANT_160
	{ 38, 2 }, // QUANT_192
	{  8, 0 }  // QUANT_256
};


/* See header for documentation. */
__inline__ __device__ unsigned int get_ise_sequence_bitcount(
	unsigned int character_count,
	quant_method quant_level
) {
	// Cope with out-of bounds values - input might be invalid
	if (static_cast<size_t>(quant_level) >= 21)
	{
		// Arbitrary large number that's more than an ASTC block can hold
		return 1024;
	}

	auto& entry = ise_sizes[quant_level];
	unsigned int divisor = (entry.divisor << 1) + 1;
	return (entry.scale * character_count + divisor - 1) / divisor;
}

struct btq_count
{
	uint8_t bits : 6;/** @brief The number of bits. */
	uint8_t trits : 1;/** @brief The number of trits. */
	uint8_t quints : 1;/** @brief The number of quints. */
};

__constant__ btq_count btq_counts[32]
{
	{ 1, 0, 0 }, // QUANT_2
	{ 0, 1, 0 }, // QUANT_3
	{ 2, 0, 0 }, // QUANT_4
	{ 0, 0, 1 }, // QUANT_5
	{ 1, 1, 0 }, // QUANT_6
	{ 3, 0, 0 }, // QUANT_8
	{ 1, 0, 1 }, // QUANT_10
	{ 2, 1, 0 }, // QUANT_12
	{ 4, 0, 0 }, // QUANT_16
	{ 2, 0, 1 }, // QUANT_20
	{ 3, 1, 0 }, // QUANT_24
	{ 5, 0, 0 }, // QUANT_32
	{ 3, 0, 1 }, // QUANT_40
	{ 4, 1, 0 }, // QUANT_48
	{ 6, 0, 0 }, // QUANT_64
	{ 4, 0, 1 }, // QUANT_80
	{ 5, 1, 0 }, // QUANT_96
	{ 7, 0, 0 }, // QUANT_128
	{ 5, 0, 1 }, // QUANT_160
	{ 6, 1, 0 }, // QUANT_192
	{ 8, 0, 0 }  // QUANT_256
};

/** @brief Packed trit values for each unpacked value, indexed [hi][][][][lo]. */
__constant__ uint8_t integer_of_trits[3][3][3][3][3]{
	{
		{
			{
				{0, 1, 2},
				{4, 5, 6},
				{8, 9, 10}
			},
			{
				{16, 17, 18},
				{20, 21, 22},
				{24, 25, 26}
			},
			{
				{3, 7, 15},
				{19, 23, 27},
				{12, 13, 14}
			}
		},
		{
			{
				{32, 33, 34},
				{36, 37, 38},
				{40, 41, 42}
			},
			{
				{48, 49, 50},
				{52, 53, 54},
				{56, 57, 58}
			},
			{
				{35, 39, 47},
				{51, 55, 59},
				{44, 45, 46}
			}
		},
		{
			{
				{64, 65, 66},
				{68, 69, 70},
				{72, 73, 74}
			},
			{
				{80, 81, 82},
				{84, 85, 86},
				{88, 89, 90}
			},
			{
				{67, 71, 79},
				{83, 87, 91},
				{76, 77, 78}
			}
		}
	},
	{
		{
			{
				{128, 129, 130},
				{132, 133, 134},
				{136, 137, 138}
			},
			{
				{144, 145, 146},
				{148, 149, 150},
				{152, 153, 154}
			},
			{
				{131, 135, 143},
				{147, 151, 155},
				{140, 141, 142}
			}
		},
		{
			{
				{160, 161, 162},
				{164, 165, 166},
				{168, 169, 170}
			},
			{
				{176, 177, 178},
				{180, 181, 182},
				{184, 185, 186}
			},
			{
				{163, 167, 175},
				{179, 183, 187},
				{172, 173, 174}
			}
		},
		{
			{
				{192, 193, 194},
				{196, 197, 198},
				{200, 201, 202}
			},
			{
				{208, 209, 210},
				{212, 213, 214},
				{216, 217, 218}
			},
			{
				{195, 199, 207},
				{211, 215, 219},
				{204, 205, 206}
			}
		}
	},
	{
		{
			{
				{96, 97, 98},
				{100, 101, 102},
				{104, 105, 106}
			},
			{
				{112, 113, 114},
				{116, 117, 118},
				{120, 121, 122}
			},
			{
				{99, 103, 111},
				{115, 119, 123},
				{108, 109, 110}
			}
		},
		{
			{
				{224, 225, 226},
				{228, 229, 230},
				{232, 233, 234}
			},
			{
				{240, 241, 242},
				{244, 245, 246},
				{248, 249, 250}
			},
			{
				{227, 231, 239},
				{243, 247, 251},
				{236, 237, 238}
			}
		},
		{
			{
				{28, 29, 30},
				{60, 61, 62},
				{92, 93, 94}
			},
			{
				{156, 157, 158},
				{188, 189, 190},
				{220, 221, 222}
			},
			{
				{31, 63, 127},
				{159, 191, 255},
				{252, 253, 254}
			}
		}
	}
};

/** @brief Packed quint values for each unpacked value, indexed [hi][mid][lo]. */
__constant__ uint8_t integer_of_quints[5][5][5]{
	{
		{0, 1, 2, 3, 4},
		{8, 9, 10, 11, 12},
		{16, 17, 18, 19, 20},
		{24, 25, 26, 27, 28},
		{5, 13, 21, 29, 6}
	},
	{
		{32, 33, 34, 35, 36},
		{40, 41, 42, 43, 44},
		{48, 49, 50, 51, 52},
		{56, 57, 58, 59, 60},
		{37, 45, 53, 61, 14}
	},
	{
		{64, 65, 66, 67, 68},
		{72, 73, 74, 75, 76},
		{80, 81, 82, 83, 84},
		{88, 89, 90, 91, 92},
		{69, 77, 85, 93, 22}
	},
	{
		{96, 97, 98, 99, 100},
		{104, 105, 106, 107, 108},
		{112, 113, 114, 115, 116},
		{120, 121, 122, 123, 124},
		{101, 109, 117, 125, 30}
	},
	{
		{102, 103, 70, 71, 38},
		{110, 111, 78, 79, 46},
		{118, 119, 86, 87, 54},
		{126, 127, 94, 95, 62},
		{39, 47, 55, 63, 31}
	}
};


__inline__ __device__ void write_bits(
	unsigned int value,
	unsigned int bitcount,
	unsigned int bitoffset,
	uint8_t* ptr
) {
	unsigned int mask = (1 << bitcount) - 1;
	value &= mask;
	int dst_byte = bitoffset >> 3;
	ptr += dst_byte;
	int sub_offset = bitoffset & 7;
	value <<= sub_offset;
	mask <<= sub_offset;
	mask = ~mask;

	int low_mask = mask & 7;
	int high_mask = (mask >> 8) & 7;

	int low_value = value;
	int high_value = value >> 8;

	//ptr[0] &= low_mask;
	ptr[0] |= low_value;
	//ptr[1] &= high_mask;
	ptr[1] |= high_value;

	int vis_low = ptr[0];
	int vis_high = ptr[1];
}

__inline__ __device__ void encode_ise(
	quant_method quant_level,
	unsigned int character_count,
	const uint8_t* input_data,
	uint8_t* output_data,
	unsigned int bit_offset
) {

	unsigned int bits = btq_counts[quant_level].bits;
	unsigned int trits = btq_counts[quant_level].trits;
	unsigned int quints = btq_counts[quant_level].quints;
	unsigned int mask = (1 << bits) - 1;

	// Write out trits and bits
	if (trits)
	{
		unsigned int i = 0;
		unsigned int full_trit_blocks = character_count / 5;

		for (unsigned int j = 0; j < full_trit_blocks; j++)
		{
			unsigned int i4 = input_data[i + 4] >> bits;
			unsigned int i3 = input_data[i + 3] >> bits;
			unsigned int i2 = input_data[i + 2] >> bits;
			unsigned int i1 = input_data[i + 1] >> bits;
			unsigned int i0 = input_data[i + 0] >> bits;

			uint8_t T = integer_of_trits[i4][i3][i2][i1][i0];

			// The max size of a trit bit count is 6, so we can always safely
			// pack a single MX value with the following 1 or 2 T bits.
			uint8_t pack;

			// Element 0 + T0 + T1
			pack = (input_data[i++] & mask) | (((T >> 0) & 0x3) << bits);
			write_bits(pack, bits + 2, bit_offset, output_data);
			bit_offset += bits + 2;

			// Element 1 + T2 + T3
			pack = (input_data[i++] & mask) | (((T >> 2) & 0x3) << bits);
			write_bits(pack, bits + 2, bit_offset, output_data);
			bit_offset += bits + 2;

			// Element 2 + T4
			pack = (input_data[i++] & mask) | (((T >> 4) & 0x1) << bits);
			write_bits(pack, bits + 1, bit_offset, output_data);
			bit_offset += bits + 1;

			// Element 3 + T5 + T6
			pack = (input_data[i++] & mask) | (((T >> 5) & 0x3) << bits);
			write_bits(pack, bits + 2, bit_offset, output_data);
			bit_offset += bits + 2;

			// Element 4 + T7
			pack = (input_data[i++] & mask) | (((T >> 7) & 0x1) << bits);
			write_bits(pack, bits + 1, bit_offset, output_data);
			bit_offset += bits + 1;
		}

		// Loop tail for a partial block
		if (i != character_count)
		{
			// i4 cannot be present - we know the block is partial
			// i0 must be present - we know the block isn't empty
			unsigned int i4 = 0;
			unsigned int i3 = i + 3 >= character_count ? 0 : input_data[i + 3] >> bits;
			unsigned int i2 = i + 2 >= character_count ? 0 : input_data[i + 2] >> bits;
			unsigned int i1 = i + 1 >= character_count ? 0 : input_data[i + 1] >> bits;
			unsigned int i0 = input_data[i + 0] >> bits;

			uint8_t T = integer_of_trits[i4][i3][i2][i1][i0];

			for (unsigned int j = 0; i < character_count; i++, j++)
			{
				// Truncated table as this iteration is always partital
				static const uint8_t tbits[4]{ 2, 2, 1, 2 };
				static const uint8_t tshift[4]{ 0, 2, 4, 5 };

				uint8_t pack = (input_data[i] & mask) |
					(((T >> tshift[j]) & ((1 << tbits[j]) - 1)) << bits);

				write_bits(pack, bits + tbits[j], bit_offset, output_data);
				bit_offset += bits + tbits[j];
			}
		}
	}
	// Write out quints and bits
	else if (quints)
	{
		unsigned int i = 0;
		unsigned int full_quint_blocks = character_count / 3;

		for (unsigned int j = 0; j < full_quint_blocks; j++)
		{
			unsigned int i2 = input_data[i + 2] >> bits;
			unsigned int i1 = input_data[i + 1] >> bits;
			unsigned int i0 = input_data[i + 0] >> bits;

			uint8_t T = integer_of_quints[i2][i1][i0];

			// The max size of a quint bit count is 5, so we can always safely
			// pack a single M value with the following 2 or 3 T bits.
			uint8_t pack;

			// Element 0
			pack = (input_data[i++] & mask) | (((T >> 0) & 0x7) << bits);
			write_bits(pack, bits + 3, bit_offset, output_data);
			bit_offset += bits + 3;

			// Element 1
			pack = (input_data[i++] & mask) | (((T >> 3) & 0x3) << bits);
			write_bits(pack, bits + 2, bit_offset, output_data);
			bit_offset += bits + 2;

			// Element 2
			pack = (input_data[i++] & mask) | (((T >> 5) & 0x3) << bits);
			write_bits(pack, bits + 2, bit_offset, output_data);
			bit_offset += bits + 2;
		}

		// Loop tail for a partial block
		if (i != character_count)
		{
			// i2 cannot be present - we know the block is partial
			// i0 must be present - we know the block isn't empty
			unsigned int i2 = 0;
			unsigned int i1 = i + 1 >= character_count ? 0 : input_data[i + 1] >> bits;
			unsigned int i0 = input_data[i + 0] >> bits;

			uint8_t T = integer_of_quints[i2][i1][i0];

			for (unsigned int j = 0; i < character_count; i++, j++)
			{
				// Truncated table as this iteration is always partital
				static const uint8_t tbits[2]{ 3, 2 };
				static const uint8_t tshift[2]{ 0, 3 };

				uint8_t pack = (input_data[i] & mask) |
					(((T >> tshift[j]) & ((1 << tbits[j]) - 1)) << bits);

				write_bits(pack, bits + tbits[j], bit_offset, output_data);
				bit_offset += bits + tbits[j];
			}
		}
	}
	// Write out just bits
	else
	{
		for (unsigned int i = 0; i < character_count; i++)
		{
			write_bits(input_data[i], bits, bit_offset, output_data);
			bit_offset += bits;
		}
	}
}
#endif