/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather ACLNN Two-Stage Interface Implementation
 *
 * Inplace: outer API takes 3 tensors (attn / lse / mask_num); the OpDef _ref
 * rename on attn_ref makes opbuild emit NnopbaseSetRef for attn, so the inner
 * API drops the duplicate attn Output param — attn is passed ONCE. lse is a
 * pure input (no SetRef, no lse output). The kernel entry receives 4 GM_ADDR
 * (attn_in, lse_in, mask_num, attn_out) with attn_in == attn_out by SetRef.
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

// Inner implementation auto-generated from OpDef. The _ref inplace rename on
// attn_ref makes opbuild emit NnopbaseSetRef for attn, so the inner API drops
// the attn Output param. lse is a pure input (no Output, no SetRef).
extern aclnnStatus aclnnInnerAlltoAllAttnUpdateAllGatherGetWorkspaceSize(
    const aclTensor *attn_ref, const aclTensor *lse_ref, const aclTensor *mask_num,
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

static bool CheckNotNull(const aclTensor *attn, const aclTensor *lse,
                         const aclTensor *mask_num)
{
    OP_CHECK_NULL(attn,     return false);
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

static bool CheckDtypeValid(const aclTensor *attn, const aclTensor *lse,
                            const aclTensor *mask_num)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(attn,     ATTN_DTYPE_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(lse,      LSE_DTYPE_LIST,  return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mask_num, MASK_DTYPE_LIST, return false);
    return true;
}

static bool CheckShape(const aclTensor *attn, const aclTensor *lse,
                       const aclTensor *mask_num)
{
    // 2D layout: attn [T·cp, n/cp·D], lse [T·cp, n/cp]
    OP_CHECK_WRONG_DIMENSION(attn, 2, return false);
    OP_CHECK_WRONG_DIMENSION(lse,  2, return false);

    // mask_num: 0-d scalar tensor (aclgraph capture requirement)
    int64_t maskDimNum = static_cast<int64_t>(mask_num->GetViewShape().GetDimNum());
    if (maskDimNum != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "mask_num must be 0-d (scalar) tensor for aclgraph; got %ld dims", maskDimNum);
        return false;
    }

    int64_t totalT    = attn->GetViewShape().GetDim(0);
    int64_t hDim      = attn->GetViewShape().GetDim(1);
    int64_t lseTotalT = lse->GetViewShape().GetDim(0);
    int64_t lseDim    = lse->GetViewShape().GetDim(1);

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

static aclnnStatus CheckParams(const aclTensor *attn, const aclTensor *lse,
                               const aclTensor *mask_num,
                               int64_t group_size)
{
    if (!CheckNotNull(attn, lse, mask_num))    return ACLNN_ERR_PARAM_NULLPTR;
    if (!CheckDtypeValid(attn, lse, mask_num)) return ACLNN_ERR_PARAM_INVALID;
    if (!CheckShape(attn, lse, mask_num))      return ACLNN_ERR_PARAM_INVALID;

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
    int64_t totalT = attn->GetViewShape().GetDim(0);
    if (totalT % group_size != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "totalT=%ld must be a multiple of group_size=%ld", totalT, group_size);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// ============================================================================
// Stage 1: GetWorkspaceSize  (inplace: outer API takes 3 tensors)
// ============================================================================

aclnnStatus aclnnAlltoAllAttnUpdateAllGatherGetWorkspaceSize(
    const aclTensor *attn,
    const aclTensor *lse,
    const aclTensor *mask_num,
    char *group,
    int64_t group_size,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();

    auto retParam = CheckParams(attn, lse, mask_num, group_size);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    // Empty tensor short-circuit: totalT == 0 means nothing to compute
    if (attn->IsEmpty()) {
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    OP_LOGD("AlltoAllAttnUpdateAllGather inplace, attn %s, lse %s, group_size=%ld",
            attn->ToString().GetString(), lse->ToString().GetString(), group_size);
    // Inplace via OpDef SetRef: pass attn ONCE (lse is a pure input, no SetRef).
    // Framework binds Output("attn_ref") slot to Input("attn_ref") same address.
    aclnnStatus ret = aclnnInnerAlltoAllAttnUpdateAllGatherGetWorkspaceSize(
        attn, lse, mask_num, group, group_size,
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
