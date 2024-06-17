#include <stdio.h>
#include <string>
#include <fstream>
#include <assert.h>
#include <vector>
#include <vector_types.h>

#include "stb_image.h"
#include "stb_image_write.h"

#include "cuda_backend.h"
#include "fgac_device_host_common.h"
#include "host\fgac_host_common.h"

#include "common\helper_math.h"

#define __device__
#define __global__
#define __constant__
#include "device\fgac_decvice_common.cuh"
#include "device\fgac_compress_symbolic.cuh"

void sim_kernel(uint8_t* dstData, uint8_t* srcData, fgac_contexti* ctx, uint32_t block_index_x, uint32_t block_index_y, uint32_t tex_size_x, uint32_t blk_num_x)
{
	float4 data_min(1e20, 1e20, 1e20, 1e20);
	float4 data_max(0, 0, 0, 0);
	float4 data_mean(0, 0, 0, 0);

	image_block img_blk;

	uint32_t dim_x = ctx->bsd.xdim;
	uint32_t dim_y = ctx->bsd.ydim;

	uint32_t tex_start_texel_pos_x = block_index_x * dim_x;
	uint32_t tex_start_texel_pos_y = block_index_y * dim_y;
	
	uint32_t dst_pos = (block_index_y * blk_num_x + block_index_x) * (128 / 8);

	int idx = 0;
	for (uint32_t index_y = 0; index_y < 4; index_y++)
	{
		for (uint32_t index_x = 0; index_x < 4; index_x++)
		{
			uint32_t texel_pos = ((tex_start_texel_pos_x + index_x) + (tex_start_texel_pos_y + index_y) * tex_size_x) * 4;

			uint8_t* data = &srcData[texel_pos];
			float4 datav = make_float4(data[0], data[1], data[2], data[3]) * (65535.0f / 255.0f);


			data_min = fminf(datav, data_min);
			data_max = fmaxf(datav, data_max);
			data_mean += datav;

			img_blk.data_r[idx] = datav.x;
			img_blk.data_g[idx] = datav.y;
			img_blk.data_b[idx] = datav.z;
			img_blk.data_a[idx] = datav.w;

			idx++;
		}
	}

	img_blk.data_min = data_min;
	img_blk.data_max = data_max;
	img_blk.data_mean = data_mean / static_cast<float>(ctx->bsd.texel_count);
	img_blk.texel_count = ctx->bsd.texel_count;

	//uint8_t pcb[16];
	compression_working_buffers tmpbuf;
	//compress_block(ctx, &img_blk, pcb, &tmpbuf);

	compress_block(ctx, &img_blk, (dstData + dst_pos), &tmpbuf);
}

void cuda_backend_test()
{
	static constexpr uint32_t blockx = 4;
	static constexpr uint32_t blocky = 4;

	std::string imagePath("G:/fgac/test/tex_test_4_4.png");
	int width = 0, height = 0, comp = 0;
	stbi_uc* srcData = stbi_load(imagePath.c_str(), &width, &height, &comp, STBI_rgb_alpha);
	uint32_t texSize = width * height * 4 * sizeof(uint8_t);

	fgac_contexti* ctx = new fgac_contexti();
	ctx->config.cw_sum_weight = 4;
	ctx->config.tune_db_limit = 2254.6614;
	init_block_size_descriptor(blockx, blocky, ctx->bsd);

	uint32_t magic = ASTC_MAGIC_ID;

	astc_header hdr;
	hdr.magic[0] = magic & 0xFF;
	hdr.magic[1] = (magic >> 8) & 0xFF;
	hdr.magic[2] = (magic >> 16) & 0xFF;
	hdr.magic[3] = (magic >> 24) & 0xFF;

	hdr.block_x = static_cast<uint8_t>(blockx);
	hdr.block_y = static_cast<uint8_t>(blocky);
	hdr.block_z = static_cast<uint8_t>(1);

	uint32_t dim_x = width;
	hdr.dim_x[0] = dim_x & 0xFF;
	hdr.dim_x[1] = (dim_x >> 8) & 0xFF;
	hdr.dim_x[2] = (dim_x >> 16) & 0xFF;

	uint32_t dim_y = height;
	hdr.dim_y[0] = dim_y & 0xFF;
	hdr.dim_y[1] = (dim_y >> 8) & 0xFF;
	hdr.dim_y[2] = (dim_y >> 16) & 0xFF;

	uint32_t dim_z = 1;
	hdr.dim_z[0] = dim_z & 0xFF;
	hdr.dim_z[1] = (dim_z >> 8) & 0xFF;
	hdr.dim_z[2] = (dim_z >> 16) & 0xFF;

	uint32_t blk_num_x = width / blockx;
	uint32_t blk_num_y = height / blocky;

	std::vector<uint8_t> outastc;
	outastc.resize(sizeof(astc_header) + 16 * blk_num_x * blk_num_y);

	memcpy(outastc.data(), &hdr, sizeof(astc_header));

	for (uint32_t indexx = 0; indexx < blk_num_x; indexx++)
	{
		for (uint32_t indexy = 0; indexy < blk_num_y; indexy++)
		{
			sim_kernel(outastc.data() + sizeof(astc_header), srcData, ctx, indexx, indexy, width, blk_num_x);
		}
	}

	FILE* fp = fopen("G:/fgac/tex_test_4_4.astc", "w");
	fwrite(outastc.data(), outastc.size(), 1, fp);
	fclose(fp);
}