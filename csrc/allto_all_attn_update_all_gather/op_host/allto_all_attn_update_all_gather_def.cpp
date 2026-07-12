/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather OpDef Registration
 *
 * Fused {AlltoAll + LSE-weighted attention update + head AllGather}, non-inplace
 * on attn along the CP (context parallel) communication group.
 *
 *   attn_in  [totalT, n_per_cp · D] bf16   (Input only - read for Phase A pack
 *                                          + inactive rows copy source)
 *   lse      [totalT, n_per_cp]     fp32   (Input only - read for Phase B weighting,
 *                                         no output: downstream does not consume it)
 *   mask_num []                     int32  (mask_per_rank; b0_total = mask_per_rank · cp_size)
 *   attn_out [totalT, n_per_cp · D] bf16   (Output - independent GM, written by
 *                                          Phase C active rows + inactive copy)
 *
 * Active rows [0, b0_total): Phase A peermem-pack -> Phase B LSE-weighted reduce
 * -> Phase C peermem head-AllGather (reverse stride) write to attn_out.
 * Inactive rows [b0_total, totalT): explicit copy attn_in -> attn_out (non-inplace,
 * no SetRef - output is an independent GM buffer, 对齐 dispatch_ffn_combine).
 *
 * 非 inplace contract: Input("attn_in") + Output("attn_out") 不同名 -> opbuild
 * 不 emit NnopbaseSetRef. lse has no Output (pure input, no SetRef).
 */

#include "register/op_def_registry.h"

namespace ops {

class AlltoAllAttnUpdateAllGather : public OpDef {
public:
    explicit AlltoAllAttnUpdateAllGather(const char *name) : OpDef(name) {
        // ===== Inputs =====
        // attn_in: 纯 input (只读) - Phase A 读 active rows + inactive rows 搬运源
        // lse:     纯 input (无 Output, 无 SetRef) - 仅 Phase B 加权读取
        this->Input("attn_in")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mask_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // ===== Outputs (非 inplace: attn_out 不同名, 无 SetRef) =====
        this->Output("attn_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // ===== Attributes =====
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("group_size").AttrType(REQUIRED).Int();

        // ===== AICore Configuration =====
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);

        // ===== MC2 Communication Domain Declaration =====
        this->MC2().HcclGroup("group");
    }
};

OP_ADD(AlltoAllAttnUpdateAllGather);

}  // namespace ops
