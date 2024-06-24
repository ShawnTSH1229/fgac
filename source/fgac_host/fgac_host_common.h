#ifndef _CUDA_HOST_COMMON_H_
#define _CUDA_HOST_COMMON_H_
#include "../fgac_common/fgac_device_host_common.h"
void init_block_descriptor(
	unsigned int block_x_texels,
	unsigned int block_y_texels,
	block_size_descriptor& bsd);

#endif