#ifndef _FGAC_SYMBOLIC_PHYSICAL_CUH_
#define _FGAC_SYMBOLIC_PHYSICAL_CUH_
#include "fgac_decvice_common.cuh"
#include "fgac_integer_sequence.cuh"
#include "fgac_weight_quant_xfer_tables.cuh"
#include "fgac_quantization.cuh"

#if ASTC_DEBUG_COUT
#include "stb_image_write.h"
#endif

// There is currently no attempt to coalesce larger void-extents
__constant__ uint8_t cbytes[8]{ 0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };

__device__ int bitrev8(int p)
{
	p = ((p & 0x0F) << 4) | ((p >> 4) & 0x0F);
	p = ((p & 0x33) << 2) | ((p >> 2) & 0x33);
	p = ((p & 0x55) << 1) | ((p >> 1) & 0x55);
	return p;
}

__device__ void symbolic_to_physical(
	const image_block& blk,
	const block_size_descriptor& bsd,
	const symbolic_compressed_block& scb,
	uint8_t pcb[16]
) 
{
	// Constant color block using UNORM16 colors
	if (scb.block_type == SYM_BTYPE_CONST_U16)
	{
		for (unsigned int i = 0; i < 8; i++)
		{
			pcb[i] = cbytes[i];
		}

		for (unsigned int i = 0; i < 4; i++)
		{
			pcb[2 * i + 8] = scb.constant_color[i] & 0xFF;
			pcb[2 * i + 9] = (scb.constant_color[i] >> 8) & 0xFF;
		}

		return;
	}

	// Compress the weights.
	// They are encoded as an ordinary integer-sequence, then bit-reversed
	uint8_t weightbuf[16]{ 0 };

	const block_mode& bm = get_block_mode(bsd,scb.block_mode);
	int weight_count = blk.texel_count;
	quant_method weight_quant_method = quant_method(bm.quant_mode);
	float weight_quant_levels = float(get_quant_level(weight_quant_method));

	const auto& qat = quant_and_xfer_tables[weight_quant_method];

	int real_weight_count = weight_count;

	int bits_for_weights = get_ise_sequence_bitcount(real_weight_count, weight_quant_method);

	uint8_t weights[64];
	{
		for (int i = 0; i < weight_count; i++)
		{
			float uqw = static_cast<float>(scb.weights[i]);
#if ASTC_DEBUG_COUT
			std::cout << "weight:" << i << " " << uqw << std::endl;
#endif
			float qw = (uqw / 64.0f) * (weight_quant_levels - 1.0f);
			int qwi = static_cast<int>(qw + 0.5f);
			weights[i] = qat.scramble_map[qwi];
		}
	}


	encode_ise(weight_quant_method, real_weight_count, weights, weightbuf, 0);

	for (int i = 0; i < 16; i++)
	{
		pcb[i] = static_cast<uint8_t>(bitrev8(weightbuf[15 - i]));
	}

	write_bits(scb.block_mode, 11, 0, pcb);
	write_bits(1 - 1, 2, 11, pcb);

	int below_weights_pos = 128 - bits_for_weights;

	{
		write_bits(scb.color_formats, 4, 13, pcb);
	}

	// Encode the color components
	uint8_t values_to_encode[32];
	int valuecount_to_encode = 0;

	const uint8_t* pack_table = color_uquant_to_scrambled_pquant_tables[scb.quant_mode - QUANT_6];
	int vals = 2 * (scb.color_formats >> 2) + 2;
	for (int j = 0; j < vals; j++)
	{
		values_to_encode[j + valuecount_to_encode] = pack_table[scb.color_values[j]];
	}
	valuecount_to_encode += vals;

	encode_ise(scb.quant_mode, valuecount_to_encode, values_to_encode, pcb, 17);

#if ASTC_DEBUG_COUT
	for (int i = 0; i < 16; i++)
	{
		std::cout << int(pcb[i]) << std::endl;;
	}

	char vis_data[4 * 4 * 4];
	int col_ifx = 0;
	int write_idx = 0;

	uint8_t vis_weights[64];
	{
		for (int i = 0; i < weight_count; i++)
		{
			float uqw = static_cast<float>(scb.weights[i]);
			float qw = (uqw / 64.0f) * (weight_quant_levels - 1.0f);
			int qwi = static_cast<int>(qw + 0.5f);
			vis_weights[i] = qwi;
		}
	};

	float4 ep0 = float4(scb.color_values[0], scb.color_values[2], scb.color_values[4], 255) * (1.0 / 255.0f);
	float4 ep1 = float4(scb.color_values[1], scb.color_values[3], scb.color_values[5], 255) * (1.0 / 255.0f);

	std::cout << "New EndPoint0:" << ep0.x << ", " << ep0.y << ", " << ep0.z << " " << std::endl;
	std::cout << "New EndPoint1:" << ep1.x << ", " << ep1.y << ", " << ep1.z << " " << std::endl;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			float4 intered_color = (ep0 + (ep1 - ep0) * float(vis_weights[col_ifx]) * (1.0 / 7.0));
			std::cout << "New Color:" << intered_color.x << ", " << intered_color.y << ", " << intered_color.z << " ";
			col_ifx++;

			vis_data[write_idx + 0] = char(int(intered_color.x * 255));
			vis_data[write_idx + 1] = char(int(intered_color.y * 255));
			vis_data[write_idx + 2] = char(int(intered_color.z * 255));
			vis_data[write_idx + 3] = char(255);
			write_idx += 4;
		}
		std::cout << std::endl;
	};
	stbi_write_tga("H:/ShawnTSH1229/fgac/tex_test_4_4_b.tga", 4, 4, 4, vis_data);
#endif
}

#endif