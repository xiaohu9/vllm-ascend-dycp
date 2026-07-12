/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather Kernel
 *
 * Per-token block-transpose AlltoAll (Phase A) → cross-cp LSE-weighted reduce
 * (Phase B) → head-AllGather + permute (Phase C). Inplace: attn_ref shares GM
 * in-out; lse is a pure input (read for weighting, no lse output — downstream
 * does not consume it). Active rows [0, b0_total) follow A→B→C; inactive rows
 * [b0_total, totalT) are left untouched (pass-through).
 *
 * Phase A routing: row r → target rank = r % cp_size_, via peermem windows.
 *   b0_total = mask_num × cp_size_  (per-rank, b0 % cp == 0).
 *   slotA row = fused [attn || lse] (Phase A still exchanges lse for Phase B).
 *   slotC row = pure attn (lse_out dropped, no lse in slotC).
 *
 * Phase B per-token math:
 *   M1 read cp peers' slotA into UB, BF16→FP32 cast (CAST_NONE)
 *   M2 ProcessLseInfReplacement +Inf → -Inf, then ReduceMax → lse_m
 *   M3 lse_exp = exp(lse_p - lse_m)
 *   M4 sum_w = Σ lse_exp
 *   M5 lse_out = lse_m + log(sum_w)        [pure intermediate, NOT written to GM]
 *   M6 norm_w = exp(lse_p - lse_out)
 *   M7 per LSE lane: load cp peers' dHead BF16 slice, cast FP32, weight, sum
 *   M8 Cast<CAST_RINT> BF16 → write slotC (peermem self window, attn only)
 *
 * Launcher: block_dim = min(aivNum, max(cp_size_, slotCRowsMax)) (batch-tailored,
 * no KERNEL_TASK_TYPE_DEFAULT). Idle cores' for-loops trivially skip but still
 * participate in every SyncAll<true>() so launcher-wide barriers gather.
 */

#pragma once

#include <limits>
#include "kernel_operator.h"
#include "allto_all_attn_update_all_gather_tiling.h"
#include "utils/moe_distribute_base.h"

namespace AlltoAllAttnUpdateAllGather {

using namespace AscendC;

// -- Static knobs --
constexpr uint32_t USED_UB_SIZE          = 160 * 1024;        // fused row ping-pong total
constexpr uint32_t USED_UB_HALF          = USED_UB_SIZE / 2;

// -- Peermem sync (生产级, 对齐 matmul_reduce_scatter_v2 + distribute_barrier) --
// flag 区起始偏移 flagOffset_ 改从 tiling 动态读取 (贴数据区尾), 不再硬编 100MB (Bug #1:
// 旧硬编 100MB 可能落到 slotA/slotC 数据区中段被 Phase A/C 数据写覆盖).
constexpr int32_t  FLAG_VALUE            = 1;                  // SetBuffFlagByAdd 单次累加值
constexpr int32_t  RAND_BASE             = 3;                  // reset 基线: reset 写 RAND_BASE, set AtomicAdd 累加, check 等值 flag+RAND_BASE (防跨 launch stale)
constexpr uint64_t CYCLES_PER_US         = 50UL;               // GetSystemCycle 计数器频率 (对齐 distribute_barrier.h:43)
constexpr uint64_t SYNC_TIMEOUT_US       = 10ULL * 1000ULL * 1000ULL;  // 10s 死锁护栏 (flag 正常 <1ms, 超时即 assert)

// Phase B numeric helpers
constexpr uint32_t NUM7           = 7;
constexpr uint32_t NUM8           = 8;
constexpr uint32_t NUM64          = 64;
constexpr uint32_t NUM256         = 256;
constexpr int32_t  ALIGNED_TO_2   = 2;
constexpr uint32_t ELEM_PER_256B  = 64;                   // = 256B / sizeof(float)
constexpr uint32_t PHASE_B_LANE_BLOCK = NUM8;             // Phase B attn merge lanes per stream block
// LSE +Inf replacement constants
static constexpr float POS_INF = std::numeric_limits<float>::infinity();
static constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
// M4/M6 -inf 守卫常量：全 -inf token (padding/残留) 时防 exp(-inf-(-inf))=nan 级联
static constexpr float NEG_FLT_MAX = std::numeric_limits<float>::lowest();   // -3.4e38 (M4-guard: lseM -inf -> -FLT_MAX)
static constexpr float FLT_MIN_VAL = std::numeric_limits<float>::min();      // 1.17e-38 (M6-guard: lseSum 0 -> FLT_MIN)

// Event IDs ∈ [0,3] per HardEvent channel (HW limit).
constexpr int32_t EV_PP_A    = 0;   // ping-pong slot 0  (MTE3_MTE2 / MTE2_MTE3)
constexpr int32_t EV_PP_B    = 1;   // ping-pong slot 1
constexpr int32_t EV_FLAG_W  = 2;   // S_MTE3      (flag write)
constexpr int32_t EV_FLAG_R  = 2;   // MTE3_MTE2   (flag read; 独立 slot, 不复用 EV_PP_B - 消除依赖 PipeBarrier 的脆弱复用)
constexpr int32_t EV_FLAG_S  = 3;   // MTE2_S      (flag readback MTE2->S)
// Phase B per-token serial sync events (no overlap with A/C ping-pong; separated by SyncAll)
constexpr int32_t EV_B_MTE2_V = 0;  // MTE2 (load slotA) → V (cast/reduce)
constexpr int32_t EV_B_V_MTE3 = 0;  // V (final cast)    → MTE3 (write slotC)
constexpr int32_t EV_B_MTE3_V = 0;  // MTE3 (last write) → V (next-token reload)

__aicore__ inline int64_t AlignUp32(int64_t x) { return (x + 31) / 32 * 32; }
__aicore__ inline uint32_t AlignUp8(uint32_t x) { return (x + NUM7) / NUM8 * NUM8; }

template <typename TilingT>
class KernelAlltoAllAttnUpdateAllGather {
public:
    __aicore__ inline KernelAlltoAllAttnUpdateAllGather(TPipe* pipe) { Ppipe = pipe; }

    __aicore__ inline void Init(
        GM_ADDR attnIn, GM_ADDR lseIn, GM_ADDR maskNum,
        GM_ADDR attnOut,
        const TilingT* tiling, GM_ADDR contextGM)
    {
        // ---- Shape / row layout (v2 共有字段) ----
        cp_size_      = tiling->groupSize;
        totalT_       = tiling->totalT;
        lseDim_       = tiling->lseDim;
        hDim_         = tiling->hDim;
        cpBatchSize_  = tiling->cpBatchSize;   // Rev 5.7: Phase B cp 流式批大小
        hAttnBytes_   = tiling->attnLineBytes;
        lseLineBytes_ = tiling->lseLineBytes;
        attnRowSize_  = tiling->attnRowSize;
        lseRowSize_   = tiling->lseRowSize;
        rowSize_      = tiling->rowSize;

        // ---- Slot layout / aivNum / tile cap ----
        aivNum_              = tiling->aivNum;
        slotABytesPerRank_   = tiling->slotABytesPerRank;
        slotCBytesPerRank_   = tiling->slotCBytesPerRank;
        slotAOffsetInWin_    = tiling->slotAOffsetInWin;     // = 0
        slotCOffsetInWin_    = tiling->slotCOffsetInWin;     // = cp_size_ · slotABytesPerRank_
        flagOffset_          = tiling->flagOffset;           // 动态 flag 区起始 (贴数据区尾)
        slotCRowsMax_        = tiling->slotCRowsMax;         // = totalT / cp_size_
        maxRowsPerSubtile_   = tiling->maxRowsPerSubtile;

        blockIdx_ = GetBlockIdx();
        // Default launcher → blockIdx_ ∈ [0, aivNum_). SplitCoreCal divides work.

        // 非 inplace: attn_in (read) 和 attn_out (write) 独立 GM (无 SetRef). lse is a
        // pure input (no SetRef) — Phase B still reads it for weighting, but no
        // lse output is written (lse_out dropped: downstream does not consume it).
        attnInGm_   = reinterpret_cast<__gm__ bfloat16_t*>(attnIn);
        lseInGm_    = reinterpret_cast<__gm__ float*>(lseIn);
        maskNumGm_  = reinterpret_cast<__gm__ int32_t*>(maskNum);
        attnOutGm_  = reinterpret_cast<__gm__ bfloat16_t*>(attnOut);   // != attnInGm_ (非 inplace)

        // Peermem window addresses — populated for all peers (cp_size_ ≤ 32 → buff_[32] enough).
        winContext_ = (__gm__ HcclOpResParam *)contextGM;
        rankId_     = winContext_->localUsrRankId;
        for (uint32_t i = 0; i < cp_size_; i++) {
            if (i == rankId_) {
                buff_[i] = (GM_ADDR)winContext_->localWindowsIn;
            } else {
                auto* remote = (HcclRankRelationResV2*)(winContext_->remoteRes[i].nextDevicePtr);
                buff_[i] = (GM_ADDR)(remote->windowsIn);
            }
        }

        // UB buffers: copyBuf_ for Pack/Unpack/Reduce ping-pong; flagBuf_ for mask_num + sync flags.
        Ppipe->InitBuffer(copyBuf_, USED_UB_SIZE);
        Ppipe->InitBuffer(flagBuf_, 64);

        // mask_num is 0-d device tensor; read at kernel runtime (aclgraph-safe).
        ReadMaskNum();
        b0_ = b0_raw_ * cp_size_;            // per-rank → total active rows
        // 生产级 fail-fast: mask_num*cp > totalT 是非法配置 (Phase A 越界写 slotA/user GM,
        // 越界读随机 lse → 比特模式被解释为 -inf/nan 级联). 不 clamp (静默截断会掩盖 bug),
        // 直接 assert. host 侧 common_cp.py 同步 raise 拦截 capture/eager, 此处拦 replay.
        assert(b0_ <= totalT_);

        ComputeTileParams();

        // Reset peermem flag area (3 flags reserved for 3 cross-rank syncs across A/B/C).
        // Only blockIdx_ == 0 writes; others wait via SyncAll.
        ResetIpcFlags(3);
        SyncAll<true>();   // 保证所有 core 的 flag 区 reset 完成 (基线 RAND_BASE 就绪) 才进 Process
    }

    __aicore__ inline void Process()
    {
        // Split work across launcher-wide AIV cores. Idle cores get count==0 and
        // for-loops empty out; they still cross every SyncAll<true>() barrier.
        SplitCoreCalForRank();
        SplitCoreCalForToken();

        if (b0_ == 0) {
            // No active token: 非 inplace 下需搬运全部 rows [0, totalT) attn_in -> attn_out
            // (inplace 下靠 SetRef pass-through; 非 inplace 下显式搬运).
            CopyInactiveRows();   // b0_=0 -> inactive=[0, totalT), 搬全部
            PipeBarrier<PIPE_ALL>();
            SyncAll<true>(); SyncAll<true>();
            SyncAll<true>(); SyncAll<true>();
            SyncAll<true>(); SyncAll<true>();
            return;
        }

        // ---- Phase A: per-token block-transpose AlltoAll (peermem) ----
        PhaseAPack();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();                       // A barrier #1
        CrossRankSyncV1(0, 1);
        SyncAll<true>();                       // A barrier #2

        // ---- Phase B: cross-cp LSE-weighted reduce (M3) ----
        PhaseBReduce();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();                       // B barrier #1
        CrossRankSyncV1(1, 1);
        SyncAll<true>();                       // B barrier #2

        // ---- Phase C: head-AllGather Combine (M4) ----
        // Pull peer slotC → 本 rank user GM,row=t·cp+srcRank 散布;cp-strided write.
        // Final cross-rank sync is still required: the next launch may overwrite slotC
        // while a slow peer is still reading it in PhaseC.
        PhaseCCombine();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();                       // C barrier #1
        CrossRankSyncV1(2, 1);
        SyncAll<true>();                       // C barrier #2

        // 非 inplace: 搬运 inactive rows [b0_total, totalT) attn_in -> attn_out
        // (inplace 下靠 SetRef pass-through; 非 inplace 下显式搬运, 规避脏数据).
        CopyInactiveRows();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();                       // inactive barrier
    }

private:
    // ====================================================================
    //  SplitCoreCal — per moe_distribute_combine.h:413-426 paradigm
    // ====================================================================
    // Phase A/C 切核单位 = rank (cp_size_ 个)
    __aicore__ inline void SplitCoreCalForRank() {
        if (aivNum_ == 0) {
            sendRankNum_ = 0; startRankId_ = 0; endRankId_ = 0;
            return;
        }
        sendRankNum_  = cp_size_ / aivNum_;
        uint32_t rem  = cp_size_ % aivNum_;
        startRankId_  = sendRankNum_ * blockIdx_;
        if (blockIdx_ < rem) {
            sendRankNum_++;
            startRankId_ += blockIdx_;
        } else {
            startRankId_ += rem;
        }
        endRankId_ = startRankId_ + sendRankNum_;
        // 当 cp_size_ < aivNum_:
        //   rem = cp_size_, sendRankNum_ baseline = 0
        //   blockIdx_ < cp_size_:  sendRankNum_=1, startRankId_=blockIdx_, endRankId_=blockIdx_+1
        //   blockIdx_ >= cp_size_: sendRankNum_=0, for-loop 空过 → 仍参与 SyncAll
    }

    // Phase B 切核单位 = active token in slotC (b0_raw_ 个 = mask_num).
    // slotCRowsMax_ 是 slot 容量上限 (totalT_/cp_size_),但 active 行只有 b0_raw_;
    // [b0_raw_, slotCRowsMax_) 区间 Phase A 不会推数据,Phase B 也不应处理。
    __aicore__ inline void SplitCoreCalForToken() {
        if (aivNum_ == 0 || b0_raw_ == 0) {
            sendTokenNum_ = 0; startTokenId_ = 0; endTokenId_ = 0;
            return;
        }
        sendTokenNum_ = b0_raw_ / aivNum_;
        uint32_t rem  = b0_raw_ % aivNum_;
        startTokenId_ = sendTokenNum_ * blockIdx_;
        if (blockIdx_ < rem) {
            sendTokenNum_++;
            startTokenId_ += blockIdx_;
        } else {
            startTokenId_ += rem;
        }
        endTokenId_ = startTokenId_ + sendTokenNum_;
    }

    // ====================================================================
    //  Phase A — per-token block-transpose AlltoAll Pack.
    //  dstRank comes from SplitCoreCalForRank's rank segment (not blockIdx_).
    // ====================================================================
    __aicore__ inline void PhaseAPack()
    {
        // Active-phase tile setup (b0_ guaranteed > 0 by caller)
        uint32_t blocksPerDst = b0_ / cp_size_;     // 每 dstRank 接收的 block 数

        for (uint32_t dstRank = startRankId_; dstRank < endRankId_; dstRank++) {
            for (uint32_t tile = 0; tile < numTiles_; tile++) {
                uint32_t tileStart = tile * maxTileB0_;
                uint32_t left      = blocksPerDst - tileStart;
                uint32_t tileB0    = left > maxTileB0_ ? maxTileB0_ : left;
                PackOneDstRank(dstRank, tileStart, tileB0);
            }
        }
        // Note: PipeBarrier outside (in Process) covers all dstRank/tile loops.
    }

    // 仿 v2 PackActiveAttnLseFused,把 (dstRank, tileStart, tileB0) 作为参数传入,
    // slot 写到 buff_[dstRank] 而不是 buff_[rankId_] —— Phase A 是把"我的对方需要的行"
    // 推到对方窗口,所以是写 buff_[dstRank]+本 rankId_ slot 偏移.
    __aicore__ inline void PackOneDstRank(uint32_t dstRank, uint32_t tileStart, uint32_t tileB0) {
        if (tileB0 == 0) return;

        // slot = dstRank 窗口里 "本 rank 的 slotA 区域" (dstRank 看到的我的行)
        GM_ADDR  slot          = buff_[dstRank] + slotAOffsetInWin_ + GetSlotABytes(rankId_);
        uint32_t attnSrcStride = (cp_size_ - 1) * hAttnBytes_;       // GM bytes (skip non-self rows)
        uint32_t lseSrcStride  = (cp_size_ - 1) * lseLineBytes_;
        uint16_t attnDstStrideBlk = (uint16_t)(lseRowSize_ / 32);    // UB 32B blocks
        uint16_t lseDstStrideBlk  = (uint16_t)(attnRowSize_ / 32);
        // 源行起点 = tileStart 个 BLOCK (cp 行 = 1 BLOCK) × cp + dstRank
        int64_t  srcRowBase    = (int64_t)tileStart * cp_size_ + dstRank;

        PpFlagInit();

        LocalTensor<uint8_t> ubAllU8 = copyBuf_.Get<uint8_t>();

        int64_t rowsLeft = tileB0;
        int64_t rowDone  = 0;
        int32_t pp = 0;
        while (rowsLeft > 0) {
            int64_t curRows = rowsLeft > (int64_t)maxRowsPerSubtile_
                ? (int64_t)maxRowsPerSubtile_
                : rowsLeft;
            int32_t ev = (pp == 0) ? EV_PP_A : EV_PP_B;
            // 真 ping-pong:切 LocalTensor 偏移到上下半 buffer (Rule §20)
            LocalTensor<uint8_t> ubBufU8 = (pp == 0) ? ubAllU8 : ubAllU8[USED_UB_HALF];

            WaitFlag<HardEvent::MTE3_MTE2>(ev);

            // (1) attn: GM→UB cp-strided read, dstStride 为 lse 段留洞
            int64_t inRow = srcRowBase + rowDone * cp_size_;
            {
                LocalTensor<bfloat16_t> ubAttn = ubBufU8.ReinterpretCast<bfloat16_t>();
                GlobalTensor<bfloat16_t> srcAttn;
                srcAttn.SetGlobalBuffer(attnInGm_ + inRow * (int64_t)hDim_);
                DataCopyExtParams rdAttn{
                    (uint16_t)curRows, hAttnBytes_, attnSrcStride, attnDstStrideBlk, 0};
                DataCopyPadExtParams<bfloat16_t> padA{false, 0, 0, 0};
                DataCopyPad(ubAttn, srcAttn, rdAttn, padA);
            }
            // (2) lse: GM→UB cp-strided read, 偏到 attnRowSize_, dstStride 为 attn 段留洞
            {
                LocalTensor<float> ubLse =
                    ubBufU8[attnRowSize_].ReinterpretCast<float>();
                GlobalTensor<float> srcLse;
                srcLse.SetGlobalBuffer(lseInGm_ + inRow * (int64_t)lseDim_);
                DataCopyExtParams rdLse{
                    (uint16_t)curRows, lseLineBytes_, lseSrcStride, lseDstStrideBlk, 0};
                DataCopyPadExtParams<float> padL{false, 0, 0, 0};
                DataCopyPad(ubLse, srcLse, rdLse, padL);
            }

            SetFlag<HardEvent::MTE2_MTE3>(ev);
            WaitFlag<HardEvent::MTE2_MTE3>(ev);

            // (3) UB→GM 单 SDMA 写 curRows·rowSize_ 字节
            int64_t dstOff = (int64_t)(tileStart + rowDone) * rowSize_;
            GlobalTensor<uint8_t> dstFused;
            dstFused.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(slot + dstOff));
            DataCopyExtParams wrParam{(uint16_t)curRows, rowSize_, 0, 0, 0};
            DataCopyPad(dstFused, ubBufU8, wrParam);

            SetFlag<HardEvent::MTE3_MTE2>(ev);

            rowsLeft -= curRows;
            rowDone  += curRows;
            pp = (pp + 1) % 2;
        }

        PpFlagFini();
    }

    // ====================================================================
    //  Phase B — per-token cross-cp LSE-weighted reduce (M3)
    //
    //  本 rank self-window 中,Phase A 完成后 slotA[i] (i ∈ [0, cp))
    //  存放 peer i 推过来的 b0_raw_ 行(active token 数 = mask_num),
    //  fused row 布局 = attn[hDim BF16] || lse[lseDim FP32]。
    //
    //  对每个 active token t ∈ [0, b0_raw_),跨 cp 个 peer 做加权求和:
    //    attn_out = Σ_i softmax(lse_i)·attn_i
    //    lse_out  = lse_max + log(Σ_i exp(lse_i − lse_max))
    //  写入本 rank slotC[t] (peermem self-window slotC 段)。
    //
    //  数学步骤参考 attention_update/decode_update.h:204-317 Compute()。
    //  这里 curLength 等价 lseDim_,curLengthPad = AlignUp8(lseDim_)。
    //
    //  LSE 全 lane 一次 reduce；attn 按 PHASE_B_LANE_BLOCK lane 流式 load/weight/sum/write。
    //  UB 只常驻 cp·PHASE_B_LANE_BLOCK·dHead,不常驻 cp·hDim,支持 lseDim > 8。
    //
    //  切核:按 b0_raw_ 切 token,每核处理 [startTokenId_, endTokenId_)。
    //  idle core (sendTokenNum_ == 0) for 循环空过,但仍参与 SyncAll。
    // ====================================================================
    __aicore__ inline void PhaseBReduce()
    {
        if (sendTokenNum_ == 0) {
            return;
        }

        const uint32_t kPad        = AlignUp8(lseDim_);
        const uint32_t spPad       = cp_size_ * kPad;
        const uint32_t spPadAlign  = ((spPad + ELEM_PER_256B - 1) / ELEM_PER_256B) * ELEM_PER_256B;
        const uint32_t dHead       = hDim_ / lseDim_;
        const uint32_t blockElems  = PHASE_B_LANE_BLOCK * dHead;
        // Rev 5.7: cp-dim batched streaming, k=cpBatchSize_ peers per batch (not cp all-in).
        const uint32_t batchElems  = cpBatchSize_ * blockElems;

        // copyBuf_ (160 KB) byte-offset segmentation. Segments do not overlap.
        // Rev 5.7: ubInFp32/ubAttnBf use batchElems (k*blockElems); added ubBcTmp (BroadCast scratch,
        //   since ubAccFp32 becomes cross-batch running sum, no longer reusable as BroadCast dst).
        LocalTensor<uint8_t> ubAll = copyBuf_.Get<uint8_t>();
        uint32_t off = 0;

        LocalTensor<float>      ubInFp32  = ubAll[off].ReinterpretCast<float>();      off += batchElems * 4;
        LocalTensor<float>      ubLseFp32 = ubAll[off].ReinterpretCast<float>();      off += spPadAlign * 4;
        LocalTensor<float>      ubLseExp  = ubAll[off].ReinterpretCast<float>();      off += spPadAlign * 4;
        LocalTensor<float>      ubLseM    = ubAll[off].ReinterpretCast<float>();      off += kPad * 4;
        LocalTensor<float>      ubLseSum  = ubAll[off].ReinterpretCast<float>();      off += kPad * 4;
        LocalTensor<float>      ubLseOut  = ubAll[off].ReinterpretCast<float>();      off += kPad * 4;
        LocalTensor<float>      ubAccFp32 = ubAll[off].ReinterpretCast<float>();      off += blockElems * 4;
        LocalTensor<float>      ubBcTmp   = ubAll[off].ReinterpretCast<float>();      off += blockElems * 4;
        LocalTensor<float>      ubNegInf  = ubAll[off].ReinterpretCast<float>();      off += spPadAlign * 4;
        LocalTensor<uint8_t>    ubMaskU8  = ubAll[off];                               off += spPadAlign;
        LocalTensor<bfloat16_t> ubAttnBf  = ubAll[off].ReinterpretCast<bfloat16_t>(); off += batchElems * 2;
        LocalTensor<bfloat16_t> ubOutBf   = ubAll[off].ReinterpretCast<bfloat16_t>(); /* off += blockElems * 2 */

        for (uint32_t t = startTokenId_; t < endTokenId_; t++) {
            ReducePerToken(t, kPad, spPad, spPadAlign,
                ubInFp32, ubLseFp32, ubLseExp,
                ubLseM, ubLseSum, ubLseOut, ubAccFp32, ubBcTmp,
                ubNegInf, ubMaskU8, ubAttnBf, ubOutBf);
            PipeBarrier<PIPE_ALL>();
        }
    }

    // 单 token 处理 — 严格对齐 decode_update.h:204-317 Compute() 数学步骤。
    __aicore__ inline void ReducePerToken(
        uint32_t t, uint32_t kPad, uint32_t spPad, uint32_t spPadAlign,
        LocalTensor<float>&      ubInFp32,
        LocalTensor<float>&      ubLseFp32,
        LocalTensor<float>&      ubLseExp,
        LocalTensor<float>&      ubLseM,
        LocalTensor<float>&      ubLseSum,
        LocalTensor<float>&      ubLseOut,
        LocalTensor<float>&      ubAccFp32,
        LocalTensor<float>&      ubBcTmp,
        LocalTensor<float>&      ubNegInf,
        LocalTensor<uint8_t>&    ubMaskU8,
        LocalTensor<bfloat16_t>& ubAttnBf,
        LocalTensor<bfloat16_t>& ubOutBf)
    {
        const int32_t cp = (int32_t)cp_size_;
        GM_ADDR selfWin = buff_[rankId_];

        // 初始化 lse 区为 NEG_INF (防止 spPad/spPadAlign padding 槽残留 +Inf 干扰 Max/Sum/Compare)
        Duplicate<float>(ubLseFp32, NEG_INF, static_cast<int32_t>(spPadAlign));
        PipeBarrier<PIPE_V>();

        // ===== M1: load cp 个 peer 在 token t 上的 LSE(FP32) =====
        for (int32_t i = 0; i < cp; i++) {
            int64_t rowOff = (int64_t)slotAOffsetInWin_
                           + (int64_t)i * (int64_t)slotABytesPerRank_
                           + (int64_t)t * (int64_t)rowSize_;
            GlobalTensor<float> srcLse;
            srcLse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                selfWin + rowOff + (int64_t)attnRowSize_));
            DataCopyExtParams rdParam{1U, lseLineBytes_, 0U, 0U, 0U};
            DataCopyPadExtParams<float> padParam{false, 0U, 0U, 0U};
            DataCopyPad(ubLseFp32[i * kPad], srcLse, rdParam, padParam);
        }
        SetFlag<HardEvent::MTE2_V>(EV_B_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EV_B_MTE2_V);

        // ===== M3: ProcessLseInfReplacement (+Inf → NEG_INF),inline =====
        {
            CompareScalar(ubMaskU8, ubLseFp32, POS_INF, CMPMODE::EQ, spPadAlign);
            PipeBarrier<PIPE_V>();
            Duplicate<float>(ubNegInf, NEG_INF, static_cast<int32_t>(spPadAlign));
            PipeBarrier<PIPE_V>();
            Select<float, uint8_t>(ubLseFp32, ubMaskU8, ubNegInf, ubLseFp32,
                SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(spPad));
            PipeBarrier<PIPE_V>();
        }

        // ===== M4: lseM = max over cp peers (8-element vector) =====
        DataCopy(ubLseM, ubLseFp32, kPad);
        PipeBarrier<PIPE_V>();
        for (int32_t i = 1; i < cp; i++) {
            Max(ubLseM, ubLseM, ubLseFp32[i * kPad], kPad);
            PipeBarrier<PIPE_V>();
        }

        // ===== M4-guard: 全 -inf token (padding/残留) 时 lseM=-inf 会使 M5 exp(-inf-(-inf))=nan =====
        // 替成 -FLT_MAX：lse(-inf) - (-FLT_MAX) = -inf，exp(-inf)=0，该 token 贡献归零不 nan。
        // 复用 ubNegInf/ubMaskU8 (M3 已完成，此处重新装载；不增加 UB 占用)。
        CompareScalar(ubMaskU8, ubLseM, NEG_INF, CMPMODE::EQ, kPad);
        PipeBarrier<PIPE_V>();
        Duplicate<float>(ubNegInf, NEG_FLT_MAX, static_cast<int32_t>(kPad));
        PipeBarrier<PIPE_V>();
        Select<float, uint8_t>(ubLseM, ubMaskU8, ubNegInf, ubLseM,
            SELMODE::VSEL_TENSOR_TENSOR_MODE, kPad);
        PipeBarrier<PIPE_V>();

        // ===== M5: lseExp = lse - lseM, then exp =====
        for (int32_t i = 0; i < cp; i++) {
            Sub(ubLseExp[i * kPad], ubLseFp32[i * kPad], ubLseM, kPad);
            PipeBarrier<PIPE_V>();
        }
        Exp(ubLseExp, ubLseExp, kPad * cp);
        PipeBarrier<PIPE_V>();

        // ===== M6: lseSum = Σ lseExp; lseOut = lseM + log(lseSum) =====
        DataCopy(ubLseSum, ubLseExp, kPad);
        PipeBarrier<PIPE_V>();
        for (int32_t i = 1; i < cp; i++) {
            Add(ubLseSum, ubLseSum, ubLseExp[i * kPad], kPad);
            PipeBarrier<PIPE_V>();
        }
        // ===== M6-guard: 全 -inf token 时 lseSum=0，log(0)=-inf 会使 lseOut=-inf，
        //                   M7 exp(-inf-(-inf))=nan。替成 FLT_MIN：log(FLT_MIN) 有限。 =====
        CompareScalar(ubMaskU8, ubLseSum, 0.0f, CMPMODE::EQ, kPad);
        PipeBarrier<PIPE_V>();
        Duplicate<float>(ubNegInf, FLT_MIN_VAL, static_cast<int32_t>(kPad));
        PipeBarrier<PIPE_V>();
        Select<float, uint8_t>(ubLseSum, ubMaskU8, ubNegInf, ubLseSum,
            SELMODE::VSEL_TENSOR_TENSOR_MODE, kPad);
        PipeBarrier<PIPE_V>();
        Log(ubLseSum, ubLseSum, kPad);
        PipeBarrier<PIPE_V>();
        Add(ubLseOut, ubLseM, ubLseSum, kPad);
        PipeBarrier<PIPE_V>();

        // ===== M7: norm_w = exp(lse - lseOut) (复用 ubLseExp) =====
        for (int32_t i = 0; i < cp; i++) {
            Sub(ubLseExp[i * kPad], ubLseFp32[i * kPad], ubLseOut, kPad);
            PipeBarrier<PIPE_V>();
        }
        Exp(ubLseExp, ubLseExp, kPad * cp);
        PipeBarrier<PIPE_V>();

        // ===== M8-M12: attn LSE lane-block streaming merge/write (avoid cp*hDim full UB resident) =====
        // Rev 5.7: cp-dim batched streaming (k=cpBatchSize_), cross-batch running sum.
        //   M1-M7 (LSE full, small) unchanged; M8-M12 attn accumulate -> cpBase step k batches:
        //   each batch load/Cast/Mul kCur peers, Add into running sum (ubAccFp32);
        //   cross-batch PipeBarrier<PIPE_ALL> ensures this batch V(Cast) finishes reading
        //   ubAttnBf/ubInFp32 before next batch MTE2(load)/V(Cast) may overwrite.
        //   Precision: accumulate order keeps peer ascending (cpBase+j), running sum 0-seed,
        //   bit-identical to old one-shot (0+x==x in IEEE754, attn non-negative).
        const uint32_t dHead = hDim_ / lseDim_;
        const uint32_t blockElems = PHASE_B_LANE_BLOCK * dHead;
        const int32_t k = (int32_t)cpBatchSize_;          // Rev 5.7: peers per batch
        GM_ADDR slotC = buff_[rankId_] + slotCOffsetInWin_ + GetSlotCBytes(rankId_);
        // slotC row is pure attn now (lse_out dropped) - stride = attnRowSize_, not rowSize_.
        int64_t dstAttnOff = (int64_t)t * (int64_t)attnRowSize_;
        int64_t localOutRow = (int64_t)t * (int64_t)cp_size_ + (int64_t)rankId_;

        for (uint32_t laneStart = 0; laneStart < lseDim_; laneStart += PHASE_B_LANE_BLOCK) {
            uint32_t laneCnt = lseDim_ - laneStart;
            laneCnt = (laneCnt > PHASE_B_LANE_BLOCK) ? PHASE_B_LANE_BLOCK : laneCnt;
            uint32_t laneElems = laneCnt * dHead;

            // running sum: first peer initializes via DataCopy (bit-identical to old one-shot
            // DataCopy[0]+Add[1..], incl. -0.0 sign - Duplicate(0)+Add(0+x) would flip -0.0 to
            // +0.0). Subsequent peers Add into running sum (cross-batch preserved).
            bool accInit = false;
            const uint32_t weightSrcShape[2] = {laneCnt, 1U};
            uint32_t weightDstShape[2] = {laneCnt, dHead};

            // cp-dim batched streaming: cpBase steps by k, kCur = min(k, cp - cpBase) peers per batch
            for (int32_t cpBase = 0; cpBase < cp; cpBase += k) {
                int32_t kCur = (cp - cpBase < k) ? (cp - cpBase) : k;

                // load kCur peers' attn block: peer index i = cpBase + j
                for (int32_t j = 0; j < kCur; j++) {
                    int32_t i = cpBase + j;
                    int64_t rowOff = (int64_t)slotAOffsetInWin_
                                   + (int64_t)i * (int64_t)slotABytesPerRank_
                                   + (int64_t)t * (int64_t)rowSize_;
                    int64_t attnLaneOff = (int64_t)laneStart * (int64_t)dHead * (int64_t)sizeof(bfloat16_t);
                    GlobalTensor<bfloat16_t> srcAttn;
                    srcAttn.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(selfWin + rowOff + attnLaneOff));
                    DataCopyExtParams rdParam{1U, laneElems * (uint32_t)sizeof(bfloat16_t), 0U, 0U, 0U};
                    DataCopyPadExtParams<bfloat16_t> padParam{false, 0, 0, 0};
                    DataCopyPad(ubAttnBf[j * blockElems], srcAttn, rdParam, padParam);
                }
                SetFlag<HardEvent::MTE2_V>(EV_B_MTE2_V);
                WaitFlag<HardEvent::MTE2_V>(EV_B_MTE2_V);

                for (int32_t j = 0; j < kCur; j++) {
                    Cast(ubInFp32[j * blockElems], ubAttnBf[j * blockElems], RoundMode::CAST_NONE, laneElems);
                }
                PipeBarrier<PIPE_V>();

                // norm_w = exp(lse - lseOut) already in ubLseExp; BroadCast to ubBcTmp, Mul into ubInFp32
                for (int32_t j = 0; j < kCur; j++) {
                    int32_t i = cpBase + j;
                    BroadCast<float, ALIGNED_TO_2, 1>(
                        ubBcTmp, ubLseExp[i * kPad + laneStart], weightDstShape, weightSrcShape);
                    PipeBarrier<PIPE_V>();
                    Mul(ubInFp32[j * blockElems], ubInFp32[j * blockElems], ubBcTmp, laneElems);
                    PipeBarrier<PIPE_V>();
                }

                // accumulate into running sum. First peer: DataCopy init (== old DataCopy[0]);
                // rest: Add. Order ubInFp32[0]+[1]+...+[cp-1] matches old one-shot bit-for-bit.
                for (int32_t j = 0; j < kCur; j++) {
                    if (!accInit) {
                        DataCopy(ubAccFp32, ubInFp32[j * blockElems], laneElems);
                        accInit = true;
                    } else {
                        Add(ubAccFp32, ubAccFp32, ubInFp32[j * blockElems], laneElems);
                    }
                    PipeBarrier<PIPE_V>();
                }

                // cross-batch barrier: this batch V(Cast) read of ubAttnBf/ubInFp32 done,
                // next batch MTE2(load)/V(Cast) may overwrite them
                PipeBarrier<PIPE_ALL>();
            }

            Cast(ubOutBf, ubAccFp32, RoundMode::CAST_RINT, laneElems);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EV_B_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EV_B_V_MTE3);

            GlobalTensor<bfloat16_t> dstAttn;
            int64_t dstLaneOff = (int64_t)laneStart * (int64_t)dHead * (int64_t)sizeof(bfloat16_t);
            DataCopyExtParams wrParam{1U, laneElems * (uint32_t)sizeof(bfloat16_t), 0U, 0U, 0U};

            // Keep slotC for peer ranks, and write this rank's final row directly.
            // PhaseCCombine skips srcRank == rankId_, saving the self peermem round-trip.
            dstAttn.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(slotC + dstAttnOff + dstLaneOff));
            DataCopyPad(dstAttn, ubOutBf, wrParam);
            SetFlag<HardEvent::MTE3_V>(EV_B_MTE3_V);
            WaitFlag<HardEvent::MTE3_V>(EV_B_MTE3_V);

            SetFlag<HardEvent::V_MTE3>(EV_B_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EV_B_V_MTE3);
            GlobalTensor<bfloat16_t> dstLocalAttn;
            dstLocalAttn.SetGlobalBuffer(attnOutGm_ + localOutRow * (int64_t)hDim_ +
                                         (int64_t)laneStart * (int64_t)dHead);
            DataCopyPad(dstLocalAttn, ubOutBf, wrParam);
            SetFlag<HardEvent::MTE3_V>(EV_B_MTE3_V);
            WaitFlag<HardEvent::MTE3_V>(EV_B_MTE3_V);
        }
        // lse_out no longer written: ubLseOut is a pure intermediate (used by M7
        // norm_w computation above). slotC row is pure attn; no lse GM writeback.
    }

    // ====================================================================
    //  Phase C — head-AllGather Combine: pull peer slotC → 本 rank user GM.
    //
    //  形态对偶 PhaseAPack:
    //    PhaseAPack   : user GM (cp-strided 读) → peer slotA (单连续写)
    //    PhaseCCombine: peer slotC (单连续读) → user GM (cp-strided 写)
    //
    //  PhaseB 把 M12 输出写到 self-window slotC[rankId_]
    //  (line 531 改为 buff_[rankId_] + slotCOffsetInWin_ + GetSlotCBytes(rankId_),
    //   GetSlotCBytes 不再含 slotCOffsetInWin_,与 GetSlotABytes 风格一致).
    //  Phase C 反向:本 rank 从 buff_[srcRank] + slotCOffsetInWin_ + GetSlotCBytes(srcRank)
    //  拉走 srcRank 算好的 b0_raw_ 行纯 attn (lse_out dropped, slotC row = attnRowSize_),
    //  按 row=t·cp+srcRank 散到本 rank attnOutGm_。srcRank == rankId_ 已由 PhaseB 直写最终输出,这里跳过。
    //
    // 非 inplace (attnOutGm_ != attnInGm_): Phase A 读 attnInGm_, Phase C 写
    //  Phase A barrier 之后 user GM 无人再读 (Phase B/C 都从 peermem 读),
    //  Phase C 的 cp-strided UB→GM 写覆盖 user GM 不与任何前序读冲突。
    // ====================================================================
    __aicore__ inline void PhaseCCombine()
    {
        uint32_t blocksPerSrc = b0_ / cp_size_;     // 每 srcRank slotC 上的 BLOCK 数

        for (uint32_t srcRank = startRankId_; srcRank < endRankId_; srcRank++) {
            if (srcRank == rankId_) {
                continue;
            }
            for (uint32_t tile = 0; tile < numTiles_; tile++) {
                uint32_t tileStart = tile * maxTileB0_;
                uint32_t left      = blocksPerSrc - tileStart;
                uint32_t tileB0    = left > maxTileB0_ ? maxTileB0_ : left;
                UnpackOneSrcRank(srcRank, tileStart, tileB0);
            }
        }
        // Note: PipeBarrier outside (in Process) covers all srcRank/tile loops.
    }

    // ====================================================================
    //  CopyInactiveRows - 非 inplace: 搬运 inactive rows [b0_total, totalT)
    //  attnInGm_ -> attnOutGm_. 经 UB 中转 (DataCopy 不支持 GM->GM, 对齐
    //  gm_ub_gm_copy.h 三段式). 按 aivNum 切 inactive rows.
    //  b0_==0 时 inactive=[0, totalT), 搬全部 rows. 复用 copyBuf_ (Phase C
    //  后/前空闲). 规避 inplace pass-through 脏数据 (SetRef 未生效时 inactive
    //  输出为垃圾) -> 非 inplace 显式搬运保证输出确定.
    // ====================================================================
    __aicore__ inline void CopyInactiveRows()
    {
        uint32_t inactiveStart = b0_;
        uint32_t inactiveEnd   = totalT_;
        if (inactiveStart >= inactiveEnd) return;   // 无 inactive rows (b0_ == totalT_)

        // 按 aivNum 切 inactive rows (对齐 SplitCoreCalForToken 切核模式)
        uint32_t totalInactive = inactiveEnd - inactiveStart;
        uint32_t perCore  = totalInactive / aivNum_;
        uint32_t rem      = totalInactive % aivNum_;
        uint32_t startRow = inactiveStart + perCore * blockIdx_;
        if (blockIdx_ < rem) {
            perCore++;
            startRow += blockIdx_;
        } else {
            startRow += rem;
        }
        uint32_t endRow = startRow + perCore;
        if (startRow >= endRow) return;   // idle core (aivNum_ > totalInactive)

        LocalTensor<uint8_t> ubU8 = copyBuf_.Get<uint8_t>();
        LocalTensor<bfloat16_t> ubAttn = ubU8.ReinterpretCast<bfloat16_t>();

        uint32_t rowsLeft = perCore;
        uint32_t rowDone  = 0;
        while (rowsLeft > 0) {
            uint32_t curRows = rowsLeft > maxRowsPerSubtile_ ? maxRowsPerSubtile_ : rowsLeft;
            uint32_t srcRow  = startRow + rowDone;
            int64_t  gmOff   = (int64_t)srcRow * (int64_t)hDim_;

            // GM -> UB (MTE2)
            GlobalTensor<bfloat16_t> srcGm;
            srcGm.SetGlobalBuffer(attnInGm_ + gmOff);
            DataCopyExtParams rd{(uint16_t)curRows, hAttnBytes_, 0, 0, 0};
            DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
            DataCopyPad(ubAttn, srcGm, rd, pad);
            SetFlag<HardEvent::MTE2_MTE3>(EV_PP_A);
            WaitFlag<HardEvent::MTE2_MTE3>(EV_PP_A);

            // UB -> GM (MTE3)
            GlobalTensor<bfloat16_t> dstGm;
            dstGm.SetGlobalBuffer(attnOutGm_ + gmOff);
            DataCopyExtParams wr{(uint16_t)curRows, hAttnBytes_, 0, 0, 0};
            DataCopyPad(dstGm, ubAttn, wr);
            SetFlag<HardEvent::MTE3_MTE2>(EV_PP_A);
            WaitFlag<HardEvent::MTE3_MTE2>(EV_PP_A);

            rowsLeft -= curRows;
            rowDone  += curRows;
        }
    }

    // 仿 v2 UnpackActiveAttnLseFused 271-338,把 (srcRank, tileStart, tileB0) 作为参数传入.
    // slot 起点用 buff_[srcRank] + slotCOffsetInWin_ + GetSlotCBytes(srcRank) (peer M12 输出).
    __aicore__ inline void UnpackOneSrcRank(uint32_t srcRank, uint32_t tileStart, uint32_t tileB0) {
        if (tileB0 == 0) return;

        // slot = srcRank 窗口里 "srcRank 自己的 slotC 段" (PhaseB 把 M12 写在那).
        // slotC row is pure attn (lse_out dropped) — read curRows·attnRowSize_ per pass.
        GM_ADDR  slot          = buff_[srcRank] + slotCOffsetInWin_ + GetSlotCBytes(srcRank);
        uint32_t attnDstStride = (cp_size_ - 1) * hAttnBytes_;       // GM bytes (skip non-self rows)
        // attnSrcStrideBlk = 0: slotC row has no lse tail, so UB rows are contiguous
        // (each row = attnRowSize_ = AlignUp32(hAttnBytes_), already 32B-padded to row boundary).
        uint16_t attnSrcStrideBlk = 0;
        // 目的行起点 = tileStart 个 BLOCK × cp + srcRank,镜像 PhaseAPack srcRowBase.
        int64_t  dstRowBase    = (int64_t)tileStart * cp_size_ + srcRank;

        PpFlagInit();

        LocalTensor<uint8_t> ubAllU8 = copyBuf_.Get<uint8_t>();

        int64_t rowsLeft = tileB0;
        int64_t rowDone  = 0;
        int32_t pp = 0;
        while (rowsLeft > 0) {
            int64_t curRows = rowsLeft > (int64_t)maxRowsPerSubtile_
                ? (int64_t)maxRowsPerSubtile_
                : rowsLeft;
            int32_t ev = (pp == 0) ? EV_PP_A : EV_PP_B;
            // 真 ping-pong: 切 LocalTensor 偏移到上下半 buffer (Rule §20)
            LocalTensor<uint8_t> ubBufU8 = (pp == 0) ? ubAllU8 : ubAllU8[USED_UB_HALF];

            WaitFlag<HardEvent::MTE3_MTE2>(ev);

            // (1) GM→UB: single SDMA read curRows·attnRowSize_ as uint8 from peer slotC
            int64_t srcOff = (int64_t)(tileStart + rowDone) * attnRowSize_;
            GlobalTensor<uint8_t> srcFused;
            srcFused.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(slot + srcOff));
            DataCopyExtParams rdParam{(uint16_t)curRows, attnRowSize_, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padParam{false, 0, 0, 0};
            DataCopyPad(ubBufU8, srcFused, rdParam, padParam);

            SetFlag<HardEvent::MTE2_MTE3>(ev);
            WaitFlag<HardEvent::MTE2_MTE3>(ev);

            // (2) attn: UB→GM, srcStride=0 (no lse tail), dst cp-strided 写
            int64_t outRow = dstRowBase + rowDone * cp_size_;
            {
                LocalTensor<bfloat16_t> ubAttn = ubBufU8.ReinterpretCast<bfloat16_t>();
                GlobalTensor<bfloat16_t> dstAttn;
                dstAttn.SetGlobalBuffer(attnOutGm_ + outRow * (int64_t)hDim_);
                DataCopyExtParams wrAttn{
                    (uint16_t)curRows, hAttnBytes_, attnSrcStrideBlk, attnDstStride, 0};
                DataCopyPad(dstAttn, ubAttn, wrAttn);
            }
            // lse_out dropped — no lse UB→GM writeback.

            SetFlag<HardEvent::MTE3_MTE2>(ev);

            rowsLeft -= curRows;
            rowDone  += curRows;
            pp = (pp + 1) % 2;
        }

        PpFlagFini();
    }

    // ====================================================================
    //  Cross-rank sync helpers (生产级, 对齐 matmul_reduce_scatter_v2 +
    //  distribute_barrier). blockIdx_==0 core owns the flag writes (launcher
    //  starts aivNum cores, so blockIdx_==0 always exists).
    //
    //  防 stale 机制: reset 写 RAND_BASE 基线 (SetBuffFlag 覆盖写 flag+RAND_BASE),
    //  notify 用 SetBuffFlagByAdd (AtomicAdd FLAG_VALUE 累加), wait 用 CheckBuffFlag
    //  等值 == flag+RAND_BASE. 跨 launch reset 覆盖时, peer 可能读到下一 launch 的
    //  set 值 (同为 RAND_BASE+FLAG_VALUE) 误判本 launch 完成 - 但误判无害:
    //  rank 进 launch N+1 的前提是其 launch N check 满足 = 所有 peer launch N Phase C
    //  已完成, 故误判不破坏数据一致性 (自己的 Phase C 在 check 前).
    // ====================================================================
    __aicore__ inline void CrossRankSyncV1(int32_t flagIdx, int32_t flagVal) {
        if (blockIdx_ == 0) {
            SetBuffFlagByAdd(reinterpret_cast<__gm__ int32_t*>(
                buff_[rankId_] + flagOffset_ + flagIdx * (int32_t)sizeof(int32_t)), FLAG_VALUE);
        }
        // Only cp_size_ cores poll peer flags. Extra launcher cores would otherwise
        // duplicate polling on the same peer via modulo and amplify peermem sync cost.
        if (blockIdx_ < cp_size_) {
            CheckBuffFlag(reinterpret_cast<__gm__ int32_t*>(
                buff_[blockIdx_] + flagOffset_ + flagIdx * (int32_t)sizeof(int32_t)),
                FLAG_VALUE * flagVal);
        }
    }

    // 覆盖写 (用于 reset): S pipe SetValue(flag+RAND_BASE) -> S_MTE3 同步 -> MTE3 DataCopy.
    // 注意 SetFlag 必须在 SetValue 之后 (S 先写 UB 再通知 MTE3 可读), 旧版 SetFlag 在
    // SetValue 前是 race bug (MTE3 可能在 SetValue 前读 UB).
    __aicore__ inline void SetBuffFlag(__gm__ int32_t *p, int32_t flag) {
        LocalTensor<int32_t> ub = flagBuf_.Get<int32_t>();
        ub.SetValue(0, flag + RAND_BASE);
        SetFlag<HardEvent::S_MTE3>(EV_FLAG_W);
        WaitFlag<HardEvent::S_MTE3>(EV_FLAG_W);
        GlobalTensor<int32_t> dst;
        dst.SetGlobalBuffer(p);
        DataCopyExtParams wrParam{1, sizeof(int32_t), 0, 0, 0};
        DataCopyPad(dst, ub, wrParam);
        PipeBarrier<PIPE_ALL>();
    }

    // AtomicAdd 累加 (用于 notify): 写裸 FLAG_VALUE, SetAtomicAdd 后 DataCopy 到 GM.
    // 对齐 matmul SetBuffFlagByAdd: 不加 RAND_BASE (累积值 = reset 基线 + 累加次数).
    __aicore__ inline void SetBuffFlagByAdd(__gm__ int32_t *p, int32_t flag) {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<int32_t> ub = flagBuf_.Get<int32_t>();
        ub.SetValue(0, flag);
        PipeBarrier<PIPE_ALL>();
        SetAtomicAdd<int32_t>();
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<int32_t> dst;
        dst.SetGlobalBuffer(p);
        DataCopyExtParams wrParam{1, sizeof(int32_t), 0, 0, 0};
        DataCopyPad(dst, ub, wrParam);
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
    }

    // 等值等待: MTE2 DataCopy(GM->UB) -> MTE2_S 同步 -> S GetValue 比较 == flag+RAND_BASE.
    // 加 GetSystemCycle 超时 assert (对齐 distribute_barrier TimeOutTest): flag 正常 <1ms,
    // 超 10s 即死锁, assert 触发 aicore 退出 (生产 fail-fast, 不静默挂死).
    __aicore__ inline void CheckBuffFlag(__gm__ int32_t *p, int32_t flag) {
        SetFlag<HardEvent::MTE3_MTE2>(EV_FLAG_R);
        WaitFlag<HardEvent::MTE3_MTE2>(EV_FLAG_R);
        LocalTensor<int32_t> ub = flagBuf_.Get<int32_t>();
        uint64_t sysBegin = static_cast<uint64_t>(GetSystemCycle());
        while (true) {
            GlobalTensor<int32_t> src;
            src.SetGlobalBuffer(p);
            DataCopyExtParams rdParam{1, sizeof(int32_t), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParam{false, 0, 0, 0};
            DataCopyPad(ub, src, rdParam, padParam);
            SetFlag<HardEvent::MTE2_S>(EV_FLAG_S);
            WaitFlag<HardEvent::MTE2_S>(EV_FLAG_S);
            if (ub.GetValue(0) == flag + RAND_BASE) {
                break;
            }
            uint64_t sysEnd = static_cast<uint64_t>(GetSystemCycle());
            uint64_t duration = (sysEnd - sysBegin) / CYCLES_PER_US;
            if (duration >= SYNC_TIMEOUT_US) {
                PipeBarrier<PIPE_ALL>();
                assert(duration < SYNC_TIMEOUT_US);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ResetIpcFlags(int32_t n) {
        if (blockIdx_ == 0) {
            for (int32_t i = 0; i < n; i++) {
                // SetBuffFlag(..., 0) 写 0+RAND_BASE = RAND_BASE 基线 (覆盖上 launch 残留).
                SetBuffFlag(reinterpret_cast<__gm__ int32_t*>(
                    buff_[rankId_] + flagOffset_ + i * (int32_t)sizeof(int32_t)), 0);
            }
        }
    }

    // ====================================================================
    //  Ping-pong handshake helpers (同 v2)
    // ====================================================================
    __aicore__ inline void PpFlagInit() {
        SetFlag<HardEvent::MTE3_MTE2>(EV_PP_A);
        SetFlag<HardEvent::MTE3_MTE2>(EV_PP_B);
    }
    __aicore__ inline void PpFlagFini() {
        WaitFlag<HardEvent::MTE3_MTE2>(EV_PP_A);
        WaitFlag<HardEvent::MTE3_MTE2>(EV_PP_B);
    }

    // ====================================================================
    //  ReadMaskNum / ComputeTileParams
    // ====================================================================
    __aicore__ inline void ReadMaskNum() {
        LocalTensor<int32_t> ub = flagBuf_.Get<int32_t>();
        GlobalTensor<int32_t> srcGT;
        srcGT.SetGlobalBuffer(maskNumGm_);
        DataCopyExtParams rdParam{1U, sizeof(int32_t), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> padParam{false, 0U, 0U, 0U};
        DataCopyPad(ub, srcGT, rdParam, padParam);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        b0_raw_ = (uint32_t)ub.GetValue(0);
    }

    __aicore__ inline void ComputeTileParams() {
        if (b0_ == 0) {
            maxTileB0_ = 0; numTiles_ = 0;
            return;
        }
        uint32_t blocksPerDst = b0_ / cp_size_;
        maxTileB0_ = blocksPerDst;
        numTiles_  = 1;
    }

    // 同 peer 的 slotA 起始字节偏移 (在某 dstRank 窗口里,本 rank 写到第 rankId_ 个 slot).
    __aicore__ inline int64_t GetSlotABytes(uint32_t peer) const {
        return (int64_t)peer * (int64_t)slotABytesPerRank_;
    }
    // slotC 起始字节偏移 (peer 段在 peermem cp 个 slotA 之后,只算 peer 的相对偏移;
    // 调用方需自行加 slotCOffsetInWin_ — 风格与 GetSlotABytes 对齐).
    __aicore__ inline int64_t GetSlotCBytes(uint32_t peer) const {
        return (int64_t)peer * (int64_t)slotCBytesPerRank_;
    }

private:
    TPipe* Ppipe = nullptr;
    __gm__ HcclOpResParam *winContext_{nullptr};
    GM_ADDR buff_[32];                              // cp_max = 32

    TBuf<TPosition::VECCALC> copyBuf_;              // ping-pong (USED_UB_SIZE)
    TBuf<TPosition::VECCALC> flagBuf_;              // 64B (mask_num + flag I/O)

    // 非 inplace: attnInGm_ (read) != attnOutGm_ (write). lse is pure input (no lse_out).
    __gm__ bfloat16_t *attnInGm_{nullptr};
    __gm__ bfloat16_t *attnOutGm_{nullptr};
    __gm__ float      *lseInGm_{nullptr};
    __gm__ int32_t    *maskNumGm_{nullptr};

    // ---- Shape / row layout (v2 共有) ----
    uint32_t blockIdx_        = 0;
    uint32_t rankId_          = 0;
    uint32_t cp_size_         = 0;
    uint32_t b0_raw_          = 0;
    uint32_t b0_              = 0;
    uint32_t totalT_          = 0;
    uint32_t lseDim_          = 0;
    uint32_t hDim_            = 0;
    uint32_t cpBatchSize_     = 0;   // Rev 5.7: Phase B cp 流式批大小(k)
    uint32_t hAttnBytes_      = 0;
    uint32_t lseLineBytes_    = 0;
    uint32_t attnRowSize_     = 0;
    uint32_t lseRowSize_      = 0;
    uint32_t rowSize_         = 0;
    uint32_t maxTileB0_       = 0;
    uint32_t numTiles_        = 0;

    // ---- slot layout / aivNum / tile cap ----
    uint32_t aivNum_              = 0;
    uint64_t slotABytesPerRank_   = 0;
    uint64_t slotCBytesPerRank_   = 0;
    uint64_t slotAOffsetInWin_    = 0;
    uint64_t slotCOffsetInWin_    = 0;
    uint64_t flagOffset_          = 0;   // 动态 flag 区起始字节偏移 (贴数据区尾, 从 tiling 读)
    uint32_t slotCRowsMax_        = 0;
    uint32_t maxRowsPerSubtile_   = 0;

    // ---- SplitCoreCal 输出 ----
    uint32_t sendRankNum_   = 0;
    uint32_t startRankId_   = 0;
    uint32_t endRankId_     = 0;
    uint32_t sendTokenNum_  = 0;
    uint32_t startTokenId_  = 0;
    uint32_t endTokenId_    = 0;
};

}  // namespace AlltoAllAttnUpdateAllGather
