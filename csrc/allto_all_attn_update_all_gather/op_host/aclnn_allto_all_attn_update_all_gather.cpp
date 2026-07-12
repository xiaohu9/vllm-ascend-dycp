/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather ACLNN Two-Stage Interface Implementation
 *
 * Non-inplace: outer API takes 4 tensors (attn_in / lse / mask_num / attn_out);
 * OpDef Input("attn_in")+Output("attn_out") distinct names -> no NnopbaseSetRef,
 * so attn_in (read) and attn_out (write) are independent GM buffers. lse is a
 * pure input (no SetRef, no lse output). The kernel entry receives 4 GM_ADDR
 * (attn_in, lse_in, mask_num, attn_out) with attn_in != attn_out; inactive rows
 * [b0_, totalT) are explicitly copied attn_in -> attn_out (CopyInactiveRows).
 *
 * NnopbaseSetHcclServerType(executor, MTE) is called before inner-execute: this
 * is a peermem-only MC2 operator (kernel drives SDMA via winContext_->localWindowsIn
 * + remoteRes[i], no hccl_.AllToAll/AllGather/AllReduce collective), so the AICPU
 * server state machine is not involved and MTE is selected.
 *
 * Inner API (aclnnInner*) is auto-generated from the OpDef.
 */

#include "aclnn_allto_all_attn_update_all_gather.h"
#include "securec.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t id;
    const char *funcName;
    bool hasReg;
} NnopbaseDfxId;

// NnopbaseSetHcclServerType second param is a strongly-typed enum (9.0.0+).
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

// Inner implementation auto-generated from OpDef (inputs, outputs, attrs order).
// 非 inplace: Input("attn_in")+Output("attn_out") 不同名 -> 无 SetRef, inner API
// receives attn_out explicitly. lse is a pure input (no Output, no SetRef).
extern aclnnStatus aclnnInnerAlltoAllAttnUpdateAllGatherGetWorkspaceSize(
    const aclTensor *attn_in, const aclTensor *lse, const aclTensor *mask_num,
    const aclTensor *attn_out,
    char *group, int64_t group_size,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerAlltoAllAttnUpdateAllGather(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);

// MC2 / nnopbase utilities (weak: NnopbaseSetHcclServerType may be unresolved
// in older nnopbase builds — caller must null-check).
extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// ===== Parameter Validation =====

static bool CheckNotNull(const aclTensor *attn_in, const aclTensor *attn_out,
                         const aclTensor *lse, const aclTensor *mask_num)
{
    OP_CHECK_NULL(attn_in,  return false);
    OP_CHECK_NULL(attn_out, return false);
    OP_CHECK_NULL(lse,      return false);
    OP_CHECK_NULL(mask_num, return false);
    return true;
}

static const std::initializer_list<op::DataType> ATTN_DTYPE_LIST = {
    op::DataType::DT_BF16
};
static const std::initializer_list<op::DataType> LSE_DTYPE_LIST = {
    op::DataType::DT_FLOAT
};
static const std::initializer_list<op::DataType> MASK_DTYPE_LIST = {
    op::DataType::DT_INT32
};

static bool CheckDtypeValid(const aclTensor *attn_in, const aclTensor *attn_out,
                            const aclTensor *lse, const aclTensor *mask_num)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(attn_in,  ATTN_DTYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(attn_out, ATTN_DTYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(lse,      LSE_DTYPE_LIST,  return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mask_num, MASK_DTYPE_LIST, return false);
    return true;
}

static bool CheckShape(const aclTensor *attn_in, const aclTensor *attn_out,
                       const aclTensor *lse, const aclTensor *mask_num)
{
    // 2D layout: attn [T·cp, n/cp·D], lse [T·cp, n/cp]
    OP_CHECK_WRONG_DIMENSION(attn_in,  2, return false);
    OP_CHECK_WRONG_DIMENSION(attn_out, 2, return false);
    OP_CHECK_WRONG_DIMENSION(lse,      2, return false);

    // mask_num: 0-d scalar tensor (aclgraph capture requirement).
    // 注意: mask_num 是 device tensor, aclnn 阶段读不到值, 无法校验 mask_num*cp > totalT.
    // 该值校验由 Python wrapper (common_cp.py raise, 基于 host int num_dycp_reqs) +
    // kernel Init assert(b0_<=totalT_) 双层镜像拦截 (覆盖 capture/eager/replay).
    int64_t maskDimNum = static_cast<int64_t>(mask_num->GetViewShape().GetDimNum());
    if (maskDimNum != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "mask_num must be 0-d (scalar) tensor for aclgraph; got %ld dims", maskDimNum);
        return false;
    }

    int64_t totalT    = attn_in->GetViewShape().GetDim(0);
    int64_t hDim      = attn_in->GetViewShape().GetDim(1);
    int64_t lseTotalT = lse->GetViewShape().GetDim(0);
    int64_t lseDim    = lse->GetViewShape().GetDim(1);

    // 非 inplace: attn_out shape 必须与 attn_in 一致 (独立 output buffer).
    int64_t outTotalT = attn_out->GetViewShape().GetDim(0);
    int64_t outHDim   = attn_out->GetViewShape().GetDim(1);
    if (outTotalT != totalT || outHDim != hDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "attn_out shape [%ld,%ld] must equal attn_in [%ld,%ld] (non-inplace output buffer)",
            outTotalT, outHDim, totalT, hDim);
        return false;
    }

    if (lseTotalT != totalT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "lse.dim(0)=%ld must equal attn.dim(0)=%ld (both = T·cp)", lseTotalT, totalT);
        return false;
    }
    if (lseDim == 0 || (hDim % lseDim) != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "attn.dim(1)=%ld must be a positive multiple of lse.dim(1)=%ld (= n/cp · D / n/cp = D)",
            hDim, lseDim);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *attn_in, const aclTensor *attn_out,
                               const aclTensor *lse, const aclTensor *mask_num,
                               int64_t group_size)
{
    if (!CheckNotNull(attn_in, attn_out, lse, mask_num))    return ACLNN_ERR_PARAM_NULLPTR;
    if (!CheckDtypeValid(attn_in, attn_out, lse, mask_num)) return ACLNN_ERR_PARAM_INVALID;
    if (!CheckShape(attn_in, attn_out, lse, mask_num))      return ACLNN_ERR_PARAM_INVALID;

    // CP sizes: 1, 2, 4, 8, 16, 32  (cp_max = 32, kernel buff_[32] hard limit;
    // 32 also bounded by peermem window capacity — see tiling window overflow check)
    if (group_size != 1 && group_size != 2 && group_size != 4 &&
        group_size != 8 && group_size != 16 && group_size != 32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "group_size must be one of {1,2,4,8,16,32}, got %ld", group_size);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // totalT must be divisible by group_size (both Phase A pack and Phase C
    // unpack walk in cp-strided rows — non-divisible totalT misaligns boundary).
    int64_t totalT = attn_in->GetViewShape().GetDim(0);
    if (totalT % group_size != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "totalT=%ld must be a multiple of group_size=%ld", totalT, group_size);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// ============================================================================
// Stage 1: GetWorkspaceSize  (non-inplace: outer API takes 4 tensors)
// ============================================================================

aclnnStatus aclnnAlltoAllAttnUpdateAllGatherGetWorkspaceSize(
    const aclTensor *attn_in,
    const aclTensor *lse,
    const aclTensor *mask_num,
    char *group,
    int64_t group_size,
    const aclTensor *attn_out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();

    auto retParam = CheckParams(attn_in, attn_out, lse, mask_num, group_size);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    // Empty tensor short-circuit: totalT == 0 means nothing to compute
    if (attn_in->IsEmpty()) {
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    OP_LOGD("AlltoAllAttnUpdateAllGather non-inplace, attn_in %s, attn_out %s, lse %s, group_size=%ld",
            attn_in->ToString().GetString(), attn_out->ToString().GetString(),
            lse->ToString().GetString(), group_size);

    // 非 inplace: pass attn_in (read) + attn_out (write) explicitly (no SetRef).
    // OpDef Input("attn_in")+Output("attn_out") 不同名 -> framework 不 bind 地址.
    aclnnStatus ret = aclnnInnerAlltoAllAttnUpdateAllGatherGetWorkspaceSize(
        attn_in, lse, mask_num, attn_out, group, group_size,
        workspaceSize, executor);
    OP_LOGD("AlltoAllAttnUpdateAllGather, aclnnInnerGetWorkspaceSize ret = %d.", ret);

    static NnopbaseDfxId dfxId = {0x60003, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

// ============================================================================
// Stage 2: Execute
// ============================================================================

aclnnStatus aclnnAlltoAllAttnUpdateAllGather(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    if (workspace == nullptr || workspaceSize == 0UL) {
        OP_LOGD("Skip api for empty tensor, workspace size %lu.", workspaceSize);
        return ACLNN_SUCCESS;
    }
    uint64_t timeStamp = NnopbaseMsprofSysTime();

    // peermem-only MC2 op → MTE server (see file header). Weak symbol: null-check.
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }

    auto ret = aclnnInnerAlltoAllAttnUpdateAllGather(workspace, workspaceSize, executor, stream);
    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "AlltoAllAttnUpdateAllGather launch aicore failed");
        return ACLNN_ERR_INNER;
    }

    static NnopbaseDfxId dfxId = {0x60003, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
