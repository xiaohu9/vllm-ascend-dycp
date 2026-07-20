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
// Flag 区偏移用真实 winContext_->winSize 贴 window 末尾 (对齐 dispatch_ffn_combine
// hccl_shmem.hpp L97/L167, 生产用真实 winSize 而非 env 估算). 修复 7e660b20 用 env 致
// mixed batch 越界 (env != 真实 winSize). winSize=0 (静态图) fallback tiling env flagOffsetBytes.
constexpr uint64_t CYCLES_PER_US   = 50ULL;
constexpr uint64_t SYNC_TIMEOUT_US = 10ULL * 1000ULL * 1000ULL;   // 10s 死锁 fail-fast
constexpr uint32_t USED_UB_SIZE          = 160 * 1024;        // fused row ping-pong total
constexpr uint32_t USED_UB_HALF          = USED_UB_SIZE / 2;

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

// Event IDs ∈ [0,3] per HardEvent channel (HW limit).
constexpr int32_t EV_PP_A    = 0;   // ping-pong slot 0  (MTE3_MTE2 / MTE2_MTE3)
constexpr int32_t EV_PP_B    = 1;   // ping-pong slot 1; EV_FLAG_R reuses MTE3_MTE2 slot 1
constexpr int32_t EV_FLAG_W  = 2;   // S_MTE3
constexpr int32_t EV_FLAG_R  = 1;   // MTE3_MTE2 (reuses EV_PP_B id; non-overlapping)
constexpr int32_t EV_FLAG_S  = 3;   // MTE2_S
// Phase B per-token serial sync events (no overlap with A/C ping-pong; separated by SyncAll)
constexpr int32_t EV_B_MTE2_V = 0;  // MTE2 (load slotA) → V (cast/reduce)
constexpr int32_t EV_B_V_MTE3 = 0;  // V (final cast)    → MTE3 (write slotC)
constexpr int32_t EV_B_MTE3_V = 0;  // MTE3 (last write) → V (next-token reload)

__aicore__ inline int64_t AlignUp32(int64_t x) { return (x + 31) / 32 * 32; }
__aicore__ inline uint32_t AlignUp8(uint32_t x) { return (x + NUM7) / NUM8 * NUM8; }
// ---- Peermem scalar sync primitives (对齐 dispatch_ffn_combine hccl_shmem.hpp L31-64) ----
// flag 同步改 scalar 直访 GM + DataCacheCleanAndInvalid, 替代旧 DataCopyPad DMA 跨 chip 读
// (旧 CheckBuffFlag reader 跨 chip peermem 读 peer window flag, DMA ordering 不保证 -> 读 stale).
// 新方案双向握手: writer gm_store 写 peer window (跨 chip) + gm_dcci clean; reader gm_load
// 读自己 window (本地) + gm_dcci invalid. reader 始终本地读, 跨 chip 只在 writer.
template <typename T>
__aicore__ inline void gm_store(__gm__ T* addr, T val) {
    *((__gm__ T*)addr) = val;
}
template <typename T>
__aicore__ inline T gm_load(__gm__ T* cache) {
    return *((__gm__ T*)cache);
}
template <typename T>
__aicore__ inline void gm_dcci(__gm__ T* addr) {
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(addr));
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}
// 轮询等 *sig_addr == cmp_val (本地读自己 window). 每轮 gm_dcci invalid 清陈旧 dcache.
// 只等 == cmp_val (不加 hccl_shmem cmp_val+1 容错: 同 launch 3 stage launchCount_ 不变,
// +1 = 下一 launch 值, 读到它说明 peer 已进下一 launch data 已被覆写 -> 误判 -> 精度错).
// 带 GetSystemCycle 10s 超时 assert (对齐 distribute_barrier.h: 死锁 fail-fast).
__aicore__ inline void gm_signal_wait(__gm__ int32_t* sig_addr, int32_t cmp_val) {
    uint64_t sysBegin = static_cast<uint64_t>(GetSystemCycle());
    while (true) {
        gm_dcci(reinterpret_cast<__gm__ uint8_t*>(sig_addr));
        if (*sig_addr == cmp_val) { return; }
        uint64_t duration = (static_cast<uint64_t>(GetSystemCycle()) - sysBegin) / CYCLES_PER_US;
        if (duration >= SYNC_TIMEOUT_US) {
            PipeBarrier<PIPE_ALL>();
            assert(duration < SYNC_TIMEOUT_US);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

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
        slotCRowsMax_        = tiling->slotCRowsMax;         // = totalT / cp_size_
        maxRowsPerSubtile_   = tiling->maxRowsPerSubtile;

        blockIdx_ = GetBlockIdx();
        // Default launcher → blockIdx_ ∈ [0, aivNum_). SplitCoreCal divides work.

        // Inplace contract: attn_in == attn_out (SetRef-guaranteed). lse is now a
        // pure input (no SetRef) — Phase B still reads it for weighting, but no
        // lse output is written (lse_out dropped: downstream does not consume it).
        attnInGm_   = reinterpret_cast<__gm__ bfloat16_t*>(attnIn);
        lseInGm_    = reinterpret_cast<__gm__ float*>(lseIn);
        maskNumGm_  = reinterpret_cast<__gm__ int32_t*>(maskNum);
        attnOutGm_  = reinterpret_cast<__gm__ bfloat16_t*>(attnOut);   // == attnInGm_

        // UB buffers: copyBuf_ for Pack/Unpack/Reduce ping-pong; flagBuf_ for mask_num + sync flags.
        // flagBuf_ 须先 Init: ReadMaskNum 依赖它 (b0==0 也需读 mask_num 判 DP).
        Ppipe->InitBuffer(copyBuf_, USED_UB_SIZE);
        Ppipe->InitBuffer(flagBuf_, 64);

        // mask_num is 0-d device tensor; read at kernel runtime (aclgraph-safe).
        ReadMaskNum();
        b0_ = b0_raw_ * cp_size_;            // per-rank → total active rows

        ComputeTileParams();

        if (b0_ > 0) {
            // Peermem window + flagOffset_ 仅 dycp(b0>0)需要; DP(b0==0)访问 winContext_ 会触发
            // MC2 peermem 懒初始化, 故从 Init 前置移入此分支.
            // Peermem window addresses - populated for all peers (cp_size_ ≤ 32 -> buff_[32] enough).
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

            // Flag 区偏移: 用真实 winContext_->winSize 贴 window 末尾 (对齐 dispatch_ffn_combine
            // hccl_shmem.hpp L97/L167, 生产用真实 winSize 而非 env 估算). 7e660b20 用 env
            // (GetMaxWindowSize 读 HCCL_BUFFSIZE) 致 mixed batch 越界: env 是配置上限, winSize 是
            // HCCL 运行时实际分配 (moe_distribute_base.h L146: 静态图可能 0, 动态图非 0), 二者可能不等.
            // winSize=0 (静态图, window 未分配) fallback tiling env flagOffsetBytes (host check 保证).
            uint64_t realWinSize = winContext_->winSize;
            uint64_t flagRegion = 64ULL + (uint64_t)cp_size_ * 192ULL;  // counter(64) + 3 stage * cp * 64
            if (realWinSize > flagRegion) {
                flagOffset_ = realWinSize - flagRegion;  // 贴真实 window 末尾
            } else {
                flagOffset_ = tiling->flagOffsetBytes;   // winSize=0 fallback: env 估算
            }

            // ---- launchCount_ 派生 + counter 更新 (对齐 dispatch_ffn_combine hccl_shmem.hpp) ----
            // counter @ flagOffset_ (每 rank 本地 window, 各 rank 独立). 读 counter+1 得本 launch
            // launchCount_ (uint32 单调, 良定义回绕 ~4.3e9 launches, 区分 launch 防 stale flag).
            // 各 core 读同一本地 counter, 值一致; 读后 SyncAll 保证都读完, 再 core0 写 counter=
            // launchCount_ 供下一 launch (读写分离防竞争). gm_dcci: 读前 invalid 清陈旧 dcache,
            // 写后 clean 刷 GM 使下一 launch 可见.
            // 仅 dycp 请求(b0>0)更新 counter: 此时 16 卡同步 +1, counter 跨 rank 一致.
            __gm__ int32_t *counter = reinterpret_cast<__gm__ int32_t*>(buff_[rankId_] + flagOffset_);
            gm_dcci(reinterpret_cast<__gm__ uint8_t*>(counter));
            launchCount_ = static_cast<uint32_t>(gm_load(counter)) + 1u;
            SyncAll<true>();   // 所有 core 读完 counter, launchCount_ 就绪
            if (blockIdx_ == 0) {
                gm_store(counter, static_cast<int32_t>(launchCount_));
                gm_dcci(reinterpret_cast<__gm__ uint8_t*>(counter));
            }
            SyncAll<true>();   // core0 counter 写完才进 Process
        }
        // b0_==0 (DP 请求): inplace pass-through(attn_out==attn_in), 无通信无 SyncAll, 直接结束.
        // counter 不更新: DP +1 会致下次 dycp launchCount_ 跨 rank 不匹配 -> CrossRankSyncV1 死锁.
    }

    __aicore__ inline void Process()
    {
        // Split work across launcher-wide AIV cores. Idle cores get count==0 and
        // for-loops empty out; they still cross every SyncAll<true>() barrier.
        SplitCoreCalForRank();
        SplitCoreCalForToken();

        if (b0_ == 0) {
            // b0_==0 (DP 请求): inplace pass-through, 直接 return, 无 SyncAll.
            return;
        }

        // ---- Phase A: per-token block-transpose AlltoAll (peermem) ----
        PhaseAPack();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();                       // A barrier #1
        CrossRankSyncV1(0);
        SyncAll<true>();                       // A barrier #2

        // ---- Phase B: cross-cp LSE-weighted reduce (M3) ----
        PhaseBReduce();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();                       // B barrier #1
        CrossRankSyncV1(1);
        SyncAll<true>();                       // B barrier #2

        // ---- Phase C: head-AllGather Combine (M4) ----
        // Pull peer slotC → 本 rank user GM,row=t·cp+srcRank 散布;cp-strided write.
        // Final cross-rank sync is still required: the next launch may overwrite slotC
        // while a slow peer is still reading it in PhaseC.
        PhaseCCombine();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();                       // C barrier #1
        CrossRankSyncV1(2);
        SyncAll<true>();                       // C barrier #2
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
    //  inplace 安全 (attnOutGm_ == attnInGm_):
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
    //  Cross-rank sync (对齐 dispatch_ffn_combine hccl_shmem.hpp CrossRankSync + mega_moe
    //  CrossRankSyncInWorldSize). 双向握手: rank R core i 写 peer i window[rankId_ slot, stage]
    //  = launchCount_ (跨 chip peermem 写, gm_store+gm_dcci clean); 等自己 window[peer i slot, stage]
    //  == launchCount_ (本地读, gm_signal_wait 每轮 gm_dcci invalid). reader 始终本地读, 跨 chip
    //  只在 writer -> 规避旧 CheckBuffFlag 跨 chip peermem 读 flag 的 DMA ordering 不保证问题.
    //  launchCount_ 单调计数器替代固定 flagVal=1 -> 区分 launch, 防 reader 读到上 launch 残留 flag
    //  误判就绪 -> 读 stale data ("输出像前次 launch" 根因).
    //
    //  flag 区布局 (每 rank window @ flagOffset_): counter @ +0 (1 int32); stage s sync 区 @
    //  +64+s*cp_size_*64, 每 rank slot 16 int32 (64B cacheline 对齐, 各 slot 独立 cacheline 防串扰).
    //  core 分摊: for i=blockIdx_; i<cp_size_; i+=aivNum_ (aivNum_>=cp_size_ 由 tiling 保证,
    //  每 core 至多 1 peer; 多余 core 空操作, 外部 SyncAll 对齐).
    //  末尾 PipeBarrier<PIPE_ALL>: 保证 S 读 flag 先于后续 MTE2 读 data (跨 pipe ordering).
    // ====================================================================
    __aicore__ inline void CrossRankSyncV1(int32_t stage) {
        int32_t count = static_cast<int32_t>(launchCount_);
        uint64_t stageBase = flagOffset_ + 64ULL
            + (uint64_t)stage * (uint64_t)cp_size_ * 64ULL;
        for (uint32_t i = blockIdx_; i < cp_size_; i += aivNum_) {
            // 1. 写 peer i window[rankId_ slot, stage] = count (跨 chip peermem 写)
            __gm__ int32_t *peerSlot = reinterpret_cast<__gm__ int32_t*>(
                buff_[i] + stageBase) + (int32_t)rankId_ * 16;
            gm_store(peerSlot, count);
            gm_dcci(reinterpret_cast<__gm__ uint8_t*>(peerSlot));
            // 2. 等自己 window[peer i slot, stage] == count (本地读)
            __gm__ int32_t *mySlot = reinterpret_cast<__gm__ int32_t*>(
                buff_[rankId_] + stageBase) + (int32_t)i * 16;
            gm_signal_wait(mySlot, count);
        }
        PipeBarrier<PIPE_ALL>();
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

    // Inplace: attnInGm_ == attnOutGm_. lse is pure input (no lse_out).
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
    uint32_t slotCRowsMax_        = 0;
    uint32_t maxRowsPerSubtile_   = 0;

    // ---- SplitCoreCal 输出 ----
    uint32_t sendRankNum_   = 0;
    uint32_t startRankId_   = 0;
    uint32_t endRankId_     = 0;
    uint32_t sendTokenNum_  = 0;
    uint32_t startTokenId_  = 0;
    uint32_t endTokenId_    = 0;

    // ---- cross-rank sync ----
    uint32_t launchCount_     = 0;
    uint64_t flagOffset_      = 0;   // 动态 flag 区偏移 (tiling 传, 贴 window 末尾)
};

}  // namespace AlltoAllAttnUpdateAllGather
