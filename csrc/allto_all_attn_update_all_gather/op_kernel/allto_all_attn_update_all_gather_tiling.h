/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather Tiling Data Structure (Kernel-side)
 *
 * Inplace: attn_ref == attn (same GM address); lse_ref == lse.
 * Active rows [0, b0_total) Phase A→B→C; inactive rows pass-through.
 *
 * CRITICAL: Mc2InitTiling and Mc2CcTiling MUST be at the front of this struct.
 *
 * b0 / b1 are derived by the kernel during Init via DataCopyPad on mask_num;
 * host tiling does not read the device 0-d tensor (aclgraph static-graph limit).
 */

#pragma once

#include "kernel_tiling/kernel_tiling.h"

namespace Mc2Tiling {

class AlltoAllAttnUpdateAllGatherTilingData {
public:
    // ===== HCCL Tiling (MUST be at front!) =====
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling   mc2CcTiling;

    // ===== Shape =====
    uint32_t totalT;           // T·cp        (attn.shape[0])
    uint32_t hDim;             // n/cp · D    (attn.shape[1])
    uint32_t lseDim;           // n/cp        (lse.shape[1])
    uint32_t groupSize;        // cp_size

    // ===== Block Dim =====
    uint32_t aivNum;           // = ascendcPlatform.GetCoreNumAiv()  (SplitCoreCal 用)

    // ===== Per-row layout (32B aligned) =====
    uint32_t attnLineBytes;    // hDim   * sizeof(bf16)
    uint32_t lseLineBytes;     // lseDim * sizeof(fp32)
    uint32_t attnRowSize;      // AlignUp32(attnLineBytes)
    uint32_t lseRowSize;       // AlignUp32(lseLineBytes)
    uint32_t rowSize;          // attnRowSize + lseRowSize

    // ===== Peermem slot layout =====
    uint32_t slotCRowsMax;             // = totalT / groupSize
    uint64_t slotABytesPerRank;        // = totalT       · rowSize        (fused [attn||lse])
    uint64_t slotCBytesPerRank;        // = slotCRowsMax · attnRowSize   (pure attn, lse_out dropped)
    uint64_t slotAOffsetInWin;         // = 0
    uint64_t slotCOffsetInWin;         // = groupSize · slotABytesPerRank

    // ===== Tile control (Phase A/C UB ping-pong) =====
    uint32_t maxRowsPerSubtile;        // single ping-pong half row 上界
    uint32_t numTiles;                 // 由 kernel runtime 根据 b0_total 派生
    uint32_t maxTileB0;                // 单 tile B0 上界

    // ===== Phase B cp-streaming (Rev 5.7) =====
    // attn cp-dim batched streaming; k peers per batch (k=cpBatchSize).
    // host adaptive k = max(1, min(cp, floor((160KB - fixedOverhead)/(blockElems*6))))
    //   keeps ubInFp32/ubAttnBf at k*blockElems (not cp*blockElems), UB <= 160KB.
    // k=cp single-batch (== old one-shot); k<cp multi-batch streaming sum.
    uint32_t cpBatchSize;              // Phase B per-batch cp count (Rev 5.7)
};

}  // namespace Mc2Tiling
