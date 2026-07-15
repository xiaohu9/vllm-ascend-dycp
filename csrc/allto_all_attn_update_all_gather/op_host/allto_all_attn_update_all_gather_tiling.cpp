/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather Host Tiling
 *
 * block_dim is batch-tailored (not a blanket aivNum):
 *   Phase A/C core upper bound = cp_size_      (SplitCoreCalForRank)
 *   Phase B   core upper bound = slotCRowsMax  (= totalT / cp_size_)
 *   block_dim = min(aivNum, max(cp_size_, slotCRowsMax))
 * Extra cores only pad SyncAll and slow the barrier, so tailoring is required.
 *
 * Tiling computes: 2D shape/attr validation, totalT % cp_size check,
 *   rowSize = AlignUp32(attnLineBytes) + AlignUp32(lseLineBytes),
 *   peermem window capacity check (slot A + slot C),
 *   Mc2CcTilingConfig (commtype = 8 = ALLTOALLV,
 *     algConfig = "AlltoAll=level0:fullmesh;level1:pairwise"),
 *   workspaces[0] = GetLibApiWorkSpaceSize(), SetScheduleMode(1).
 * numTiles / maxTileB0 are derived by the kernel at runtime (mask_num is device-only).
 */

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "error_log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

#include "../op_kernel/allto_all_attn_update_all_gather_tiling.h"

// dev error_log.h spells this TILIING (three I's); provide the TILING alias so the
// tiling body (written against the standard spelling) compiles unchanged.
#ifndef VECTOR_INNER_ERR_REPORT_TILING
#define VECTOR_INNER_ERR_REPORT_TILING(op_name, err_msg, ...) \
    OP_LOGE(op_name, err_msg, ##__VA_ARGS__)
#endif

namespace {
// 与 dispatch_ffn_combine_tiling.cpp::GetMaxWindowSize 行为对齐.
// HCCL_BUFFSIZE 单位 MB,默认 200MB.
constexpr const char* HCCL_BUFFSIZE_ENV = "HCCL_BUFFSIZE";
constexpr uint64_t MB_TO_BYTES = 1024ULL * 1024ULL;
constexpr uint64_t DEFAULT_HCCL_BUFFSIZE_MB = 200ULL;

static uint64_t GetMaxWindowSize() {
    uint64_t windowSizeMb = DEFAULT_HCCL_BUFFSIZE_MB;
    const char* env = std::getenv(HCCL_BUFFSIZE_ENV);
    if (env != nullptr) {
        try {
            uint64_t v = std::stoull(env);
            if (v > 0 && v <= std::numeric_limits<uint16_t>::max()) {
                windowSizeMb = v;
            }
        } catch (const std::exception&) {
            // ignore — fallback to default
        }
    }
    return windowSizeMb * MB_TO_BYTES;
}
}  // namespace

using namespace AscendC;
using namespace ge;

namespace optiling {

// 32B-align (peermem slot stride requirement)
static inline int64_t AlignUp32(int64_t bytes) {
    return (bytes + 31) / 32 * 32;
}

// Phase A/C ping-pong 单半 = 80KB; 单 subtile 装 MAX_ROWS_PER_SUBTILE=2 行
// → rowSize ≤ 80KB / 2 = 40KB. (kernel.h USED_UB_SIZE = 160 KB)
static constexpr int64_t MAX_ROW_SIZE_BYTES         = 40 * 1024;

// Phase B per-token reduce UB 总容量上界 = USED_UB_SIZE = 160 KB.
// Layout (一次性方案,见 kernel.h PhaseBReduce):
//   lsePad=AlignUp8(lseDim), spPadAlign=AlignUp64(cp·lsePad), dHead=hDim/lseDim
//   LSE 全 lane 一次 reduce；attn 按 PHASE_B_LANE_BLOCK lane 流式 merge/write。
//   ubInFp32 cp·laneBlock·dHead·4 | ubLseFp32/ubLseExp/ubNegInf spPadAlign·4 each
//   | ubLseM/Sum/Out lsePad·4·3 | ubAccFp32 laneBlock·dHead·4
//   | ubMaskU8 spPadAlign | ubAttnBf cp·laneBlock·dHead·2 | ubOutBf laneBlock·dHead·2
static constexpr int64_t MAX_REDUCE_UB_BYTES        = 160 * 1024;
static constexpr int64_t PHASE_B_LANE_BLOCK         = 8;

// 单 subtile 行数（与 v2 USED_UB_HALF / (2·rowSize) 一致；hardcode 2）
static constexpr uint32_t MAX_ROWS_PER_SUBTILE      = 2;

static ge::graphStatus AlltoAllAttnUpdateAllGatherTilingFunc(gert::TilingContext *context) {
    auto tilingData = context->GetTilingData<Mc2Tiling::AlltoAllAttnUpdateAllGatherTilingData>();

    // ---------- 1. Tensor shapes ----------
    const gert::StorageShape *attnShape = context->GetInputShape(0);   // [totalT, n/cp·D]
    const gert::StorageShape *lseShape  = context->GetInputShape(1);   // [totalT, n/cp]
    OP_TILING_CHECK(attnShape == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "attn_ref shape is nullptr"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(lseShape == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "lse_ref shape is nullptr"),
        return ge::GRAPH_FAILED);

    const auto &attnStorage = attnShape->GetStorageShape();
    const auto &lseStorage  = lseShape->GetStorageShape();
    OP_TILING_CHECK(attnStorage.GetDimNum() != 2,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "attn_ref must be 2D, got %zu dims", attnStorage.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(lseStorage.GetDimNum() != 2,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "lse_ref must be 2D, got %zu dims", lseStorage.GetDimNum()),
        return ge::GRAPH_FAILED);

    uint32_t totalT = static_cast<uint32_t>(attnStorage.GetDim(0));
    uint32_t hDim   = static_cast<uint32_t>(attnStorage.GetDim(1));
    uint32_t lseDim = static_cast<uint32_t>(lseStorage.GetDim(1));

    OP_TILING_CHECK(static_cast<uint32_t>(lseStorage.GetDim(0)) != totalT,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "lse_ref.dim(0)=%u must equal attn_ref.dim(0)=%u",
            static_cast<uint32_t>(lseStorage.GetDim(0)), totalT),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(lseDim == 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "lse_ref.dim(1) must be > 0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hDim % lseDim != 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "attn_ref.dim(1)=%u must be a multiple of lse_ref.dim(1)=%u (D)",
            hDim, lseDim),
        return ge::GRAPH_FAILED);

    // ---------- 2. Attrs ----------
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "attrs is nullptr"),
        return ge::GRAPH_FAILED);
    auto groupPtr     = attrs->GetAttrPointer<char>(0);
    auto groupSizePtr = attrs->GetAttrPointer<int64_t>(1);
    OP_TILING_CHECK(groupPtr == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "group attr is nullptr"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(groupSizePtr == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "group_size attr is nullptr"),
        return ge::GRAPH_FAILED);

    uint32_t groupSize = static_cast<uint32_t>(*groupSizePtr);
    OP_TILING_CHECK(groupSize == 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "group_size must be > 0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(totalT % groupSize != 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "totalT=%u must be a multiple of group_size=%u (CP)", totalT, groupSize),
        return ge::GRAPH_FAILED);

    // ---------- 3. Tiling struct (shape) ----------
    tilingData->totalT        = totalT;
    tilingData->hDim          = hDim;
    tilingData->lseDim        = lseDim;
    tilingData->groupSize     = groupSize;
    tilingData->attnLineBytes = hDim   * static_cast<uint32_t>(sizeof(uint16_t));   // bf16
    tilingData->lseLineBytes  = lseDim * static_cast<uint32_t>(sizeof(float));      // fp32
    tilingData->attnRowSize   = static_cast<uint32_t>(AlignUp32(tilingData->attnLineBytes));
    tilingData->lseRowSize    = static_cast<uint32_t>(AlignUp32(tilingData->lseLineBytes));
    tilingData->rowSize       = tilingData->attnRowSize + tilingData->lseRowSize;

    OP_TILING_CHECK(static_cast<int64_t>(tilingData->rowSize) > MAX_ROW_SIZE_BYTES,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "rowSize=%u exceeds Phase A/C UB-per-subtile cap %ld "
            "(attnRow=%u + lseRow=%u). Reduce hDim or lseDim.",
            tilingData->rowSize, MAX_ROW_SIZE_BYTES,
            tilingData->attnRowSize, tilingData->lseRowSize),
        return ge::GRAPH_FAILED);

    // Phase B per-token reduce UB capacity check (LSE full-lane + attn lane-streaming, see kernel.h PhaseBReduce)
    // Rev 5.7: cp-dim batched streaming (k=cpBatchSize). UB footprint governed by k, always <= MAX_REDUCE_UB_BYTES.
    //   k = max(1, min(cp, floor((MAX_REDUCE_UB_BYTES - fixedOverhead)/perPeerBytes)))
    //   perPeerBytes   = blockElems*6 (ubInFp32*4 + ubAttnBf*2)
    //   fixedOverhead  = LSE seg (spPadAlign*4*3 + lsePad*4*3 + spPadAlign)
    //                   + attn seg (ubAccFp32*4 + ubBcTmp*4 + ubOutBf*2 = blockElems*10)
    //   k=cp -> single batch (== old one-shot); k<cp -> multi-batch streaming sum.
    int64_t cp_int     = static_cast<int64_t>(groupSize);
    int64_t hDim_int   = static_cast<int64_t>(hDim);
    int64_t lseDim_int = static_cast<int64_t>(lseDim);
    int64_t lsePad_int = ((lseDim_int + 7) / 8) * 8;
    int64_t dHead_int  = hDim_int / lseDim_int;
    int64_t laneBlock_int = PHASE_B_LANE_BLOCK;
    int64_t blockElems_int = laneBlock_int * dHead_int;
    int64_t spPad_int      = cp_int * lsePad_int;
    int64_t spPadAlign_int = ((spPad_int + 63) / 64) * 64;

    const int64_t perPeerBytes     = blockElems_int * 6;        // ubInFp32*4 + ubAttnBf*2
    const int64_t lseFixedOverhead = spPadAlign_int * 4 * 3     // ubLseFp32 + ubLseExp + ubNegInf
                                   + lsePad_int * 4 * 3         // ubLseM + ubLseSum + ubLseOut
                                   + spPadAlign_int;            // ubMaskU8
    const int64_t attnFixedOverhead = blockElems_int * 4        // ubAccFp32
                                    + blockElems_int * 4        // ubBcTmp (Rev 5.7: BroadCast scratch)
                                    + blockElems_int * 2;       // ubOutBf
    const int64_t fixedOverhead = lseFixedOverhead + attnFixedOverhead;

    int64_t kMax = (MAX_REDUCE_UB_BYTES - fixedOverhead) / perPeerBytes;
    if (kMax < 1) {
        kMax = 1;
    }
    int64_t k = (cp_int < kMax) ? cp_int : kMax;                // k in [1, cp]
    uint32_t cpBatchSize = static_cast<uint32_t>(k);

    int64_t reduceUbBytes = k * perPeerBytes + fixedOverhead;
    OP_TILING_CHECK(reduceUbBytes > MAX_REDUCE_UB_BYTES,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "Phase B reduce UB bytes %ld > USED_UB_SIZE %ld "
            "(cp=%ld, k(cpBatchSize)=%u, hDim=%ld, lseDim=%ld, lsePad=%ld, spPadAlign=%ld, dHead=%ld, laneBlock=%ld). "
            "Internal error: k-adaptive formula should guarantee fit. Reduce hDim/lseDim/laneBlock.",
            reduceUbBytes, MAX_REDUCE_UB_BYTES, cp_int, cpBatchSize, hDim_int, lseDim_int,
            lsePad_int, spPadAlign_int, dHead_int, laneBlock_int),
        return ge::GRAPH_FAILED);

    tilingData->cpBatchSize = cpBatchSize;                      // Rev 5.7: Phase B cp streaming batch size

    // ---------- 4. Peermem slot 布局 ----------
    // slot A: cp 个 rank 区段，每段 totalT 行（按 mask_num 上界 b0_total ≤ totalT 预留）
    //         slotA 行 = fused [attn || lse] (Phase A 仍交换 lse 给 Phase B 加权) → stride = rowSize
    // slot C: cp 个 rank 区段，每段 totalT/cp 行（head-AllGather 后每 rank 仅持自己的子表）
    //         slotC 行 = 纯 attn (lse_out 已去除, 下游不消费) → stride = attnRowSize
    tilingData->slotCRowsMax        = totalT / groupSize;
    tilingData->slotABytesPerRank   = (uint64_t)totalT * tilingData->rowSize;
    tilingData->slotCBytesPerRank   = (uint64_t)tilingData->slotCRowsMax * tilingData->attnRowSize;
    tilingData->slotAOffsetInWin    = 0ULL;
    tilingData->slotCOffsetInWin    = (uint64_t)groupSize * tilingData->slotABytesPerRank;

    int64_t neededWinBytes = static_cast<int64_t>(
        groupSize * tilingData->slotABytesPerRank +
        groupSize * tilingData->slotCBytesPerRank);
    uint64_t maxWinBytes = GetMaxWindowSize();
    OP_TILING_CHECK(static_cast<uint64_t>(neededWinBytes) > maxWinBytes,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "peermem window overflow: needed %ld > max %lu "
            "(cp=%u totalT=%u rowSize=%u slotA/rk=%lu slotC/rk=%lu). "
            "Increase HCCL_BUFFSIZE or reduce D/totalT.",
            neededWinBytes, maxWinBytes, groupSize, totalT,
            tilingData->rowSize,
            (unsigned long)tilingData->slotABytesPerRank,
            (unsigned long)tilingData->slotCBytesPerRank),
        return ge::GRAPH_FAILED);

    // ---------- 4b. flag 区动态偏移 (Rev 5.8, 对齐 dispatch_ffn_combine hccl_shmem.hpp) ----------
    // flag 区贴 window 末尾: flagOffsetBytes = maxWinBytes - FLAG_REGION_BYTES. data 区
    // [0, neededWinBytes), flag 区 [flagOffsetBytes, maxWinBytes). 合一 check: 不重叠即
    // neededWinBytes + FLAG_REGION_BYTES <= maxWinBytes (数据不侵入 flag + flag 在 window 内).
    // flag 区 = counter(64B) + 3 stage * cp_size_ * 64B = 64 + cp*192; 64B cacheline 对齐
    // (maxWinBytes MB 对齐, FLAG_REGION_BYTES 64 倍数). 替代旧硬编 100MB, 解除数据区 100MB 上限.
    // 各 rank flagOffsetBytes 一致 (HCCL_BUFFSIZE + cp_size 均同) -> 跨 rank 同步地址对齐.
    const uint64_t FLAG_REGION_BYTES = 64ULL + (uint64_t)groupSize * 192ULL;
    OP_TILING_CHECK(static_cast<uint64_t>(neededWinBytes) + FLAG_REGION_BYTES > maxWinBytes,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "peermem window overflow: needed %ld + flagRegion %lu > maxWin %lu "
            "(cp=%u totalT=%u rowSize=%u). Reduce D/totalT or increase HCCL_BUFFSIZE.",
            neededWinBytes, (unsigned long)FLAG_REGION_BYTES, (unsigned long)maxWinBytes,
            groupSize, totalT, tilingData->rowSize),
        return ge::GRAPH_FAILED);
    tilingData->flagOffsetBytes = maxWinBytes - FLAG_REGION_BYTES;

    // ---------- 5. HCCL window allocation (no collective is ever issued) ----------
    // commtype = ALLTOALLV(8); A3 拓扑(910_93) HCCL 内部按 group 自动识别,
    // algConfig 字符串与邻居 dispatch_ffn_combine 同款.
    // 注:删除原 SetCommEngine(AIV_ENGINE) 调用 — vllm-ascend csrc 邻居均无此调用,
    //    Mc2CcTilingConfig 默认引擎已满足.
    std::string algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    uint32_t commtype = 8U;  // HCCL_CMD_ALLTOALLV
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupPtr, commtype, algConfig);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "GetTiling mc2InitTiling failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "GetTiling mc2CcTiling failed"),
        return ge::GRAPH_FAILED);

    // ---------- 6. Workspace ----------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "get workspace failed"),
        return ge::GRAPH_FAILED);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    // ---------- 7. Block dim batch-tailored ----------
    //   Phase A/C core upper bound = cp_size_     (SplitCoreCalForRank: cp dst ranks)
    //   Phase B   core upper bound = slotCRowsMax (SplitCoreCalForToken: ≤ totalT/cp)
    //   block_dim = min(aivNum, max(cp_size_, slotCRowsMax))
    // Extra cores only pad SyncAll and slow the barrier.
    uint32_t aivNum   = ascendcPlatform.GetCoreNumAiv();
    uint32_t needRank = groupSize;
    uint32_t needTok  = tilingData->slotCRowsMax;
    uint32_t needMax  = (needRank > needTok) ? needRank : needTok;
    uint32_t usedAiv  = (needMax < aivNum) ? needMax : aivNum;

    OP_TILING_CHECK(aivNum < groupSize,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "aivNum=%u must be >= groupSize=%u", aivNum, groupSize),
        return ge::GRAPH_FAILED);

    tilingData->aivNum            = usedAiv;
    tilingData->maxRowsPerSubtile = MAX_ROWS_PER_SUBTILE;
    // numTiles / maxTileB0 由 kernel Init 在读取 mask_num 后派生 — host 无法读 device 0-d tensor.
    tilingData->numTiles  = 0;
    tilingData->maxTileB0 = 0;

    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(usedAiv, 0, usedAiv);
    context->SetBlockDim(blockDim);

    // SetScheduleMode(1) batch mode — 必须，因 SyncAll 需 launcher 等齐全部核.
    context->SetScheduleMode(1);

    OP_LOGD(context->GetNodeName(),
        "AlltoAllAttnUpdateAllGather tiling: totalT=%u hDim=%u lseDim=%u groupSize=%u "
        "attnRow=%u lseRow=%u rowSize=%u slotA/rk=%lu slotC/rk=%lu "
        "physAivNum=%u usedAiv=%u (needRank=%u needTok=%u) blockDim=%u",
        totalT, hDim, lseDim, groupSize,
        tilingData->attnRowSize, tilingData->lseRowSize, tilingData->rowSize,
        (unsigned long)tilingData->slotABytesPerRank,
        (unsigned long)tilingData->slotCBytesPerRank,
        aivNum, usedAiv, needRank, needTok, blockDim);
    return ge::GRAPH_SUCCESS;
}

struct AlltoAllAttnUpdateAllGatherCompileInfo {};

static ge::graphStatus TilingParseForAlltoAllAttnUpdateAllGather(
    [[maybe_unused]] gert::TilingParseContext *context) {
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AlltoAllAttnUpdateAllGather)
    .Tiling(AlltoAllAttnUpdateAllGatherTilingFunc)
    .TilingParse<AlltoAllAttnUpdateAllGatherCompileInfo>(TilingParseForAlltoAllAttnUpdateAllGather);

}  // namespace optiling
