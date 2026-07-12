/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * AlltoAllAttnUpdateAllGather InferShape & InferDataType
 *
 * 非 inplace: Input("attn_in") + Output("attn_out") 不同名 -> 无 NnopbaseSetRef.
 * Only attn_out is an output (lse is now a pure input — no lse output).
 * Output shape = input shape — the kernel only rearranges rows + does
 * LSE-weighted reduce + head-AllGather, so tensor shape is unchanged.
 */

#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShapeAlltoAllAttnUpdateAllGather(gert::InferShapeContext* context) {
    auto attnShape = context->GetOutputShape(0);    // attn_out [totalT, n_per_cp · D]

    if (context->GetInputShape(0) == nullptr || attnShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Output shape matches input shape exactly (non-inplace, no shape change).
    *attnShape = *context->GetInputShape(0);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAlltoAllAttnUpdateAllGather(gert::InferDataTypeContext* context) {
    // attn_out: BF16 (same as Input attn_in)
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AlltoAllAttnUpdateAllGather)
    .InferShape(InferShapeAlltoAllAttnUpdateAllGather)
    .InferDataType(InferDataTypeAlltoAllAttnUpdateAllGather);

}  // namespace ops
