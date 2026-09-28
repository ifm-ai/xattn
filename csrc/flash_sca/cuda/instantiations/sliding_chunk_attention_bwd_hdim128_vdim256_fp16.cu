#include "flash_sca/cuda/template.cuh"
#include "flash_sca/cuda/instantiate.h"

XATTN_FLASH_SCA_BWD_CASE_INSTANTIATE(FP16, at::Half, 128, 256)
