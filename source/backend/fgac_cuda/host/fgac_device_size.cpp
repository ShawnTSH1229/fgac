#include <array>
#include "fgac_host_common.h"

struct ise_size
{
	/** @brief The scaling parameter. */
	uint8_t scale : 6;

	/** @brief The divisor parameter. */
	uint8_t divisor : 2;
};

// e.g. RANGE 80 = 4, 0, 1 : range = 5 * 2 ^ 4
// bit num = n * s + alignup( 7 * s / 3) n = 4 s = count
static const std::array<ise_size, 21> ise_sizes{ {
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
} };

/* See header for documentation. */
unsigned int get_ise_sequence_bitcounts(
	unsigned int character_count,
	quant_method quant_level
) {
	if (static_cast<size_t>(quant_level) >= ise_sizes.size())
	{
		return 1024;
	}

	auto& entry = ise_sizes[quant_level];
	unsigned int divisor = (entry.divisor << 1) + 1;
	return (entry.scale * character_count + divisor - 1) / divisor;
}

static bool decode_block_mode_2d(
	unsigned int block_mode,
	unsigned int& x_weights,
	unsigned int& y_weights,
	bool& is_dual_plane,
	unsigned int& quant_mode,
	unsigned int& weight_bits
)
{
	x_weights = 0;
	y_weights = 0;

	uint8_t R0 = (block_mode >> 4) & 1;
	uint8_t H = (block_mode >> 9) & 1;
	uint8_t D = (block_mode >> 10) & 1;
	uint8_t A = (block_mode >> 5) & 0x3;

	if ((block_mode & 3) != 0) // case bit 0 and bit 1 equal to R0 and R1
	{
		R0 |= (block_mode & 3) << 1; // R1 and R1
		unsigned int B = (block_mode >> 7) & 3;
		switch ((block_mode >> 2) & 3)
		{
		case 0:
			x_weights = B + 4;
			y_weights = A + 2;
			break;
		case 1:
			x_weights = B + 8;
			y_weights = A + 2;
			break;
		case 2:
			x_weights = A + 2;
			y_weights = B + 8;
			break;
		case 3:
			B &= 1;
			if (block_mode & 0x100)
			{
				x_weights = B + 2;
				y_weights = A + 2;
			}
			else
			{
				x_weights = A + 2;
				y_weights = B + 6;
			}
			break;
		}
	}
	else
	{
		R0 |= ((block_mode >> 2) & 3) << 1;
		if (((block_mode >> 2) & 3) == 0)
		{
			return false;
		}

		unsigned int B = (block_mode >> 9) & 3;
		switch ((block_mode >> 7) & 3)
		{
		case 0:
			x_weights = 12;
			y_weights = A + 2;
			break;
		case 1:
			x_weights = A + 2;
			y_weights = 12;
			break;
		case 2:
			x_weights = A + 6;
			y_weights = B + 6;
			D = 0;
			H = 0;
			break;
		case 3:
			switch ((block_mode >> 5) & 3)
			{
			case 0:
				x_weights = 6;
				y_weights = 10;
				break;
			case 1:
				x_weights = 10;
				y_weights = 6;
				break;
			case 2:
			case 3:
				return false;
			}
			break;
		}
	}

	unsigned int weight_count = x_weights * y_weights * (D + 1);
	quant_mode = (R0 - 2) + 6 * H;
	is_dual_plane = D != 0;
	weight_bits = get_ise_sequence_bitcounts(weight_count, static_cast<quant_method>(quant_mode));
	return (weight_count <= BLOCK_MAX_WEIGHTS && weight_bits >= BLOCK_MIN_WEIGHT_BITS && weight_bits <= BLOCK_MAX_WEIGHT_BITS);

}

static void construct_block_size_descriptor_2d(
	unsigned int x_texels,
	unsigned int y_texels,
	block_size_descriptor& bsd)
{
	bsd.xdim = x_texels;
	bsd.ydim = y_texels;
	bsd.texel_count = x_texels * y_texels;

	for (unsigned int i = 0; i < WEIGHTS_MAX_BLOCK_MODES; i++)
	{
		bsd.block_mode_packed_index[i] = BLOCK_BAD_BLOCK_MODE;
	}

	unsigned int packed_bm_idx = 0;

	for (unsigned int i = 0; i < WEIGHTS_MAX_BLOCK_MODES; i++)
	{
		// Skip modes we've already included in a previous pass
		if (bsd.block_mode_packed_index[i] != BLOCK_BAD_BLOCK_MODE)
		{
			continue;
		}

		// Decode parameters
		unsigned int x_weights;
		unsigned int y_weights;
		bool is_dual_plane;
		unsigned int quant_mode;
		unsigned int weight_bits;
		bool valid = decode_block_mode_2d(i, x_weights, y_weights, is_dual_plane, quant_mode, weight_bits);

		// Always skip invalid encodings for the current block size
		if (!valid || (x_weights > x_texels) || (y_weights > y_texels))
		{
			continue;
		}

		if (is_dual_plane)
		{
			continue;
		}

		if ((111 - weight_bits) <= 0)
		{
			continue;
		}

		auto& bm = bsd.block_modes[packed_bm_idx];

		bm.quant_mode = static_cast<uint8_t>(quant_mode);
		bm.weight_bits = static_cast<uint8_t>(weight_bits);
		bm.mode_index = static_cast<uint16_t>(i);

		bsd.block_mode_packed_index[i] = static_cast<uint16_t>(packed_bm_idx);

		packed_bm_idx++;
	}

	bsd.block_mode_count_1plane_selected = packed_bm_idx;
}

void init_block_size_descriptor(
	unsigned int block_x_texels,
	unsigned int block_y_texels,
	block_size_descriptor& bsd
)
{
	construct_block_size_descriptor_2d(block_x_texels, block_y_texels, bsd);
}