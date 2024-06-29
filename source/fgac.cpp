#include <stdio.h>
#include <string>
#include <fstream>
#include <assert.h>
#include <vector>
#include <vector_types.h>

#include "stb_image.h"
#include "stb_image_write.h"
#include "fgac_common/fgac_device_host_common.h"
#include "fgac_host/fgac_host_common.h"
#include "cxxopts.hpp"

extern "C" void image_compress(uint8_t * dstData, const uint8_t* const srcData, const block_size_descriptor* const bsd, uint32_t tex_size_x, uint32_t tex_size_y, uint32_t blk_num_x, uint32_t blk_num_y, uint32_t dest_offset);

int main(int argc, char* argv[])
{
	cxxopts::Options opts("FGAC", "fast gpu astc compression");
	opts.allow_unrecognised_options();
	opts.add_options()
		("i,input_image", "Input image path", cxxopts::value<std::string>())
		("o,output_astc", "Output astc path", cxxopts::value<std::string>())
		("h,help", "Print help message.");

	auto opt_result = opts.parse(argc, argv);
	if (opt_result.count("h") || argc < 2 || opt_result.count("i") == 0 || opt_result.count("o") == 0)
	{
		printf(opts.help().c_str());
		printf("\n");
		printf("example: -i H:/xxx/xxx/test_img_0.jpeg -o H:/xxx/xxx/tex_test_out.astc");
		exit(-1);
	}

	std::string input_image_path = opt_result["i"].as<std::string>();
	std::string output_astc_path = opt_result["o"].as<std::string>();


	static constexpr uint32_t blockx = 4;
	static constexpr uint32_t blocky = 4;

	int width = 0, height = 0, comp = 0;
	stbi_uc* srcData = stbi_load(input_image_path.c_str(), &width, &height, &comp, STBI_rgb_alpha);
	uint32_t texSize = width * height * 4 * sizeof(uint8_t);

	uint32_t blk_num_x = width / blockx;
	uint32_t blk_num_y = height / blocky;

	std::vector<uint8_t> outastc;
	outastc.resize(sizeof(astc_header) + 16 * blk_num_x * blk_num_y);

	astc_header* hdr = (astc_header*)outastc.data();
	uint32_t magic = ASTC_MAGIC_ID;
	hdr->magic[0] = magic & 0xFF;
	hdr->magic[1] = (magic >> 8) & 0xFF;
	hdr->magic[2] = (magic >> 16) & 0xFF;
	hdr->magic[3] = (magic >> 24) & 0xFF;

	hdr->block_x = static_cast<uint8_t>(blockx);
	hdr->block_y = static_cast<uint8_t>(blocky);
	hdr->block_z = static_cast<uint8_t>(1);

	uint32_t dim_x = width;
	hdr->dim_x[0] = dim_x & 0xFF;
	hdr->dim_x[1] = (dim_x >> 8) & 0xFF;
	hdr->dim_x[2] = (dim_x >> 16) & 0xFF;

	uint32_t dim_y = height;
	hdr->dim_y[0] = dim_y & 0xFF;
	hdr->dim_y[1] = (dim_y >> 8) & 0xFF;
	hdr->dim_y[2] = (dim_y >> 16) & 0xFF;

	uint32_t dim_z = 1;
	hdr->dim_z[0] = dim_z & 0xFF;
	hdr->dim_z[1] = (dim_z >> 8) & 0xFF;
	hdr->dim_z[2] = (dim_z >> 16) & 0xFF;

	block_size_descriptor bsd;
	init_block_descriptor(blockx, blocky, bsd);

	image_compress(outastc.data(), srcData, &bsd, width, height, blk_num_x, blk_num_y, sizeof(astc_header));

	std::ofstream file(output_astc_path, std::ios::out | std::ios::binary);
	file.write((char*)outastc.data(), outastc.size());

	return 0;
}
