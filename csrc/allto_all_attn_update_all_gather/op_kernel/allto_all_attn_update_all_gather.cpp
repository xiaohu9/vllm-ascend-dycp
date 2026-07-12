/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather Kernel Entry
 *
 * Non-inplace contract via OpDef Input("attn_in")+Output("attn_out") distinct
 * names -> no NnopbaseSetRef -> framework allocates attn_out as an independent
 * GM buffer (not bound to attn_in).
 *   attn_in            -> pure input (Phase A reads active rows [0, b0_);
 *                        CopyInactiveRows reads inactive rows [b0_, totalT)).
 *   attn_out           -> pure output (Phase C writes [0, b0_); CopyInactiveRows
 *                        writes [b0_, totalT) - distinct empty buffer must be
 *                        filled explicitly, SetRef pass-through no longer exists).
 *   lse_in             -> pure input (no SetRef, no lse output - dropped: the
 *                        fused op only needs lse for Phase B weighting, not as
 *                        an output; downstream does not consume it).
 * Inner API generated from OpDef passes 4 tensors (attn_in, lse input,
 * mask_num, attn_out); kernel entry receives 4 GM_ADDR (attn_in, lse_in,
 * mask_num, attn_out - attn_in != attn_out). mask_num is a 0-d int32 device
 * tensor - kernel reads it at runtime via DataCopyPad (aclgraph-compatible).
 *
 * Launcher: default (NO KERNEL_TASK_TYPE_DEFAULT). Host SetBlockDim =
 * min(aivNum, max(cp_size_, slotCRowsMax)). SplitCoreCalFor{Rank,Token}
 * divides work; idle AIV-core for-loops trivially skip but still participate
 * in launcher-wide SyncAll barriers (no bare return per F17).
 */

#include "kernel_operator.h"
#include "allto_all_attn_update_all_gather_tiling.h"
#include "allto_all_attn_update_all_gather_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void allto_all_attn_update_all_gather(
    GM_ADDR attn_in, GM_ADDR lse_in, GM_ADDR mask_num,
    GM_ADDR attn_out,                                      // 非 inplace: != attn_in
    GM_ADDR workspace, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(Mc2Tiling::AlltoAllAttnUpdateAllGatherTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    TPipe pipe;
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();

    AlltoAllAttnUpdateAllGather::KernelAlltoAllAttnUpdateAllGather<
        Mc2Tiling::AlltoAllAttnUpdateAllGatherTilingData> op(&pipe);
    op.Init(attn_in, lse_in, mask_num, attn_out, &tilingData, contextGM);
    op.Process();
}
