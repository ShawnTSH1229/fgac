#ifndef _FGAC_DEVICE_COMMON_H_
#define _FGAC_DEVICE_COMMON_H_
#include "../fgac_common/fgac_device_host_common.h"
#include "../fgac_common/helper_math.h"

__shared__ float4 block_src_tex[BLOCK_MAX_TEXELS];
__shared__ float block_reduce_shared[BLOCK_MAX_TEXELS];

#endif
