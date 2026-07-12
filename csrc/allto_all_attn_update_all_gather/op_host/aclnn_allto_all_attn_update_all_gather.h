/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather ACLNN Two-Stage Interface Declaration
 *
 * Non-inplace fused {alltoall + cross-cp LSE-weighted attn update + head-AllGather
 * + permute} for CP (context parallel). OpDef Input("attn_in")+Output("attn_out")
 * distinct names -> no NnopbaseSetRef; attn_in (read) and attn_out (write) are
 * independent GM buffers. lse is a pure input (no SetRef, no lse output - the
 * fused op only needs lse for Phase B weighting, not as an output).
 *
 *   attn_in            : [T*cp, n/cp * D]  bf16   (Input only - read)
 *   attn_out           : [T*cp, n/cp * D]  bf16   (Output only - written)
 *   lse                : [T*cp, n/cp]      fp32   (Input only - no output)
 *   mask_num           : []  int32  (0-d; per-rank active token count)
 *   group, group_size  : HCCL group descriptor (group_size = cp in {1,2,4,8,16,32})
 *
 * Active rows [0, mask_num x cp): undergo full three-phase fusion -> attn_out.
 * Inactive rows [mask_num x cp, T*cp): explicitly copied attn_in -> attn_out
 * (CopyInactiveRows, since non-inplace attn_out is a distinct empty buffer).
 */

#pragma once

#include "aclnn/aclnn_base.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Stage 1: Compute workspace size and validate parameters.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllAttnUpdateAllGatherGetWorkspaceSize(
    const aclTensor *attn_in,
    const aclTensor *lse,
    const aclTensor *mask_num,
    char *group,
    int64_t group_size,
    const aclTensor *attn_out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * Stage 2: Execute.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllAttnUpdateAllGather(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif
