# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import triton.language as tl
import numpy as np
import sys
import os
from typing import Any, Callable, Dict, Optional, Tuple
import aiter
from aiter.test_common import checkAllclose, perftest
from aiter import pertoken_quant
from aiter.fused_moe_gelu import fused_topk
from aiter.fused_moe_bf16_asm import asm_moe, torch_moe, moe_sorting_ck
from op_tests.int4_utils import *

# @perftest(num_iters=3)
def torch_moe_stage1(hidden_states,
                     w1,  # E, inter_dim*2, model_dim
                     w2,  # E, model_dim, inter_dim
                     topk_weight, topk_ids,
                     dtype=torch.float16,
                     # following for quant
                     fc1_scale=None,  # [expert, inter_dim, 1]
                     w1_scale=None,  # [1]
                     a1_scale=None,  # [expert]]
                     block_size=32
                     ):
    ctype = torch.float  # compute type
    hidden_states = hidden_states.to(ctype)
    w1 = w1.to(ctype)

    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    N = w1.shape[1]
    num_experts, model_dim, inter_dim = w2.shape
    hidden_states = hidden_states.view(
        B, -1, D).repeat(1, topk, 1)

    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk

    # gose to quant D_w8a8/w8a8
    if fc1_scale is not None:
        w1 = (w1.view(-1, D) * fc1_scale.view(-1, 1)).view(num_experts, -1, D)
    if a1_scale is not None and w1_scale is not None:
        hidden_states = hidden_states * a1_scale
        w1 = w1 * w1_scale.view(-1, 1, 1)

    out = torch.zeros(
        (B, topk, N),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            out[mask] = act_input

    return out.to(dtype)


# @perftest(num_iters=3)
def torch_moe_stage2(hidden_states,
                     w1,  # E, inter_dim*2, model_dim
                     w2,  # E, model_dim, inter_dim
                     topk_weights, topk_ids,
                     sorted_weights, sorted_ids,
                     sorted_expert_ids, num_valid_ids,
                     dtype=torch.float16,
                     w2_scale=None,  # [1]
                     a2_scale=None,  # [expert]]
                     block_size=32
                     ):
    ctype = torch.float  # compute type
    hidden_states = hidden_states.to(ctype)
    w2 = w2.to(ctype)

    token_num, topk = topk_ids.shape
    # M, _ = hidden_states.shape
    num_experts, model_dim, inter_dim = w2.shape
    max_num_m_blocks = sorted_expert_ids.shape[0]

    # gose to quant D_w8a8/w8a8
    if a2_scale is not None and w2_scale is not None:
        hidden_states = hidden_states * a2_scale
        w2 = w2 * w2_scale.view(-1, 1, 1)

    out = torch.zeros(
        (token_num, topk, model_dim),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w2[E_id].transpose(0, 1))
            out[mask] = act_input
    return (out * topk_weights.view(token_num, -1, 1)).sum(1).to(dtype)


def torch_moe(hidden_states, w1, w2, topk_weight, topk_ids,
              # following for quant
              fc1_scale=None,  # [expert, inter_dim, 1]
              fc2_scale=None,  # [expert, model_dim, 1]
              fc1_smooth_scale=None,  # [expert, 1, model_dim]
              fc2_smooth_scale=None,  # [expert, 1, inter_dim]
              ):
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    dtype = hidden_states.dtype
    hidden_states = hidden_states.view(
        B, -1, D).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, D),
        dtype=dtype,
        device=hidden_states.device,
    )
    # g1u1(w1 include gate and up)
    if w2.shape[2]*2 == w1.shape[1]:
        moeType = "g1u1"
        inter_dim = w2.shape[2]
    # g1u0(w1 only include gate)
    else:
        moeType = "g1u0"
        inter_dim = w1.shape[1]
    # gose to quant D_w8a8/w8a8
    if fc1_scale is not None:
        expert = w1.shape[0]
        w2D = w2.shape[-1]
        w1 = (w1.view(-1, D).to(fc1_scale) *
              fc1_scale.view(-1, 1)).to(dtype).view(expert, -1, D)
        w2 = (w2.view(-1, w2D).to(fc2_scale) *
              fc2_scale.view(-1, 1)).to(dtype).view(expert, -1, w2D)
    if fc1_smooth_scale is not None:
        expert = fc1_smooth_scale.shape[0]
        fc1_smooth_scale = fc1_smooth_scale.view(expert, -1).to(dtype)
        fc2_smooth_scale = fc2_smooth_scale.view(expert, -1).to(dtype)

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            if fc1_smooth_scale is not None:
                sub_tokens = sub_tokens * (
                    fc1_smooth_scale[E_id])
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            if moeType == "g1u1":
                gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
                act_out = F.silu(gate) * up
            else:
                act_out = F.gelu(act_input)
            if fc2_smooth_scale is not None:
                act_out = act_out * (
                    fc2_smooth_scale[E_id])
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (
        out * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


# @perftest()
def ck_moe_stage1(hidden_states,
                  w1,  # [E, inter_dim*2, model_dim]
                  w2,  # [E, model_dim, inter_dim]
                  sorted_token_ids,  # [max_num_tokens_padded]
                  sorted_expert_ids,  # [max_num_m_blocks]
                  num_valid_ids,  # [1]
                  w1_scale, a1_scale, dtype,
                  topk,
                  block_size=32
                  ):
    token_num = hidden_states.shape[0]
    D = w1.shape[1]
    num_experts, model_dim, inter_dim = w2.shape
    max_num_tokens_padded = sorted_token_ids.shape[0]
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    out = torch.zeros(
        (token_num, topk, D),
        dtype=dtype,
        device=hidden_states.device,
    )
    aiter.ck_moe_stage1(hidden_states, w1, w2, sorted_token_ids,
                        sorted_expert_ids, num_valid_ids, out, topk, w1_scale, a1_scale, block_size)
    return out


@perftest()
def ck_moe_stage2(hidden_states,
                  w1,  # [E, inter_dim*2, model_dim]
                  w2,  # [E, model_dim, inter_dim]
                  sorted_token_ids,  # [max_num_tokens_padded]
                  sorted_expert_ids,  # [max_num_m_blocks]
                  sorted_weights,  # [max_num_tokens_padded]
                  num_valid_ids,  # [1]
                  w2_scale, a2_scale, dtype,
                  topk,
                  block_size=32
                  ):
    token_num = hidden_states.shape[0]
    D = w2.shape[1]
    num_experts, model_dim, inter_dim = w2.shape
    max_num_tokens_padded = sorted_token_ids.shape[0]
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    out = torch.zeros(
        (token_num, D),
        dtype=dtype,
        device=hidden_states.device,
    )
    aiter.ck_moe_stage2(hidden_states, w1, w2, sorted_token_ids,
                        sorted_expert_ids, sorted_weights,
                        num_valid_ids, out, topk, w2_scale, a2_scale, block_size)
    return out


# @perftest()
# def ck_moe_fused_2stages(hidden_states,
#                          # [expert(local_expert:EP), inter_dim(*2), dim] N,K
#                          w1,
#                          w2,  # [expert(local_expert:EP), dim, inter_dim]
#                          topk_weight, topk_ids,
#                          # following for int8 quant
#                          # [expert(local_expert:EP), inter_dim, 1]
#                          fc1_scale=None,
#                          # [expert(local_expert:EP), model_dim, 1]
#                          fc2_scale=None,
#                          block_size=32,
#                          a1_scale=None
#                          ):
#     return ck_moe_2stages(hidden_states, w1, w2, topk_weight, topk_ids,
#                           fc1_scale, fc2_scale, block_size=block_size, a1_scale=a1_scale)

import torch

def generate_repeated_tensor(input_tensor):
    shape = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype
    cnt = 32
    # sequence = torch.tensor([0,0,0,0,2,0,0,0,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4], dtype=dtype, device=device)
    sequence = torch.linspace(-1, 1, cnt, dtype=dtype, device=device)
    print(sequence)
    total_elements = torch.prod(torch.tensor(shape)).item()  
    repeat_times = total_elements  // cnt  

    repeated_tensor = sequence.repeat(repeat_times)[:total_elements]

    repeated_tensor = repeated_tensor.reshape(shape)
    return repeated_tensor

def shuffle_weight222(x: torch.Tensor, layout=(32, 32)) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    IN, IK = layout
    BK = IK*2
    K = 32
    BN = IN
    assert (x.shape[-2] %
            BN == 0), f'{x.shape[-2]} % {BN} == {x.shape[-2] % BN }'
    assert (x.shape[-1] %
            BK == 0), f'{x.shape[-1]} % {BK} == {x.shape[-1] % BK }'

    x_ = x
    x_ = x_.view(-1,
                 x.shape[-2]//BN, BN,
                 x.shape[-1]//BK, BK//K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    return x_


def test_fmoe(dtype, token, model_dim, inter_dim, E, topk, quant='No', use_g1u1=False, shared_E=0):
    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    if use_g1u1:
        w1 = torch.randn((E+shared_E, inter_dim*2, model_dim),
                         dtype=dtype, device="cuda") / 10
    else:
        w1 = torch.randn((E+shared_E, inter_dim, model_dim),
                         dtype=dtype, device="cuda")
    w2 = torch.randn((E+shared_E, model_dim, inter_dim),
                     dtype=dtype, device="cuda")
    score = torch.randn((token, E), device="cuda", dtype=dtype)
    
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    E, model_dim, inter_dim = w2.shape
    M, topk = topk_ids.shape
    BLOCK_SIZE_M = 128
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting_ck(topk_ids, topk_weights, E,
                                                                                           model_dim, dtype, BLOCK_SIZE_M)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    quant_dtype = torch.float8_e4m3fnuz
    quant_dtype_w = torch.int8
    # w1 = torch.ones_like(w1)
    # w2 = torch.ones_like(w2)
    w1_qt, w1_scale = aiter.pertoken_quant(w1.view(E, -1),
                                           quant_dtype=quant_dtype_w, dtypeMax=7)
    w2_qt, w2_scale = aiter.pertoken_quant(w2.view(E, -1),
                                           quant_dtype=quant_dtype_w, dtypeMax=7)
    a1_qt, a1_scale = aiter.per_tensor_quant(input,  quant_dtype=quant_dtype, dtypeMax=7)
    w1_qt = w1_qt.view(w1.shape)
    w2_qt = w2_qt.view(w2.shape)
    sp1 = (E+shared_E, inter_dim)
    sp2 = (E+shared_E, model_dim)
    # a1_qt = torch.ones_like(a1_qt)
    # w1_qt = torch.ones_like(w1_qt)
    # w1_qt = generate_repeated_tensor(w1_qt)
    # w1_scale = torch.ones_like(w1_scale)
    # a1_scale = torch.ones_like(a1_scale)
    # b implement
    w1b = rearrange_4bit_elements(convert_int8_to_uint32_int4(shuffle_weight222(w1_qt)))
    w2b = rearrange_4bit_elements(convert_int8_to_uint32_int4(shuffle_weight222(w2_qt)))
    # print("a1:",a1_qt, a1_scale)
    # print("w1:",w1_qt, w1_scale)
    # print("topk:", topk_weights)
    out1_ref = torch_moe_stage1(a1_qt, w1_qt,
                                        w2_qt,
                                        topk_weights, topk_ids,
                                        dtype=dtype,
                                        fc1_scale=None,
                                        w1_scale=w1_scale,
                                        a1_scale=a1_scale,
                                        block_size=BLOCK_SIZE_M)
    # print(out1_ref)
    if use_g1u1:
        gate, up = out1_ref.split([inter_dim, inter_dim], dim=-1)
        input2 = F.silu(gate) * up
    else:
        input2 = F.gelu(out1_ref)
    a2_qt, a2_scale = aiter.per_tensor_quant(input2,  quant_dtype=quant_dtype)
    # a2_qt, a2_scale = aiter.per_tensor_quant_fp8_hip(input2)
    out2_ref = torch_moe_stage2(a2_qt,
                                        w1_qt,  # E, inter_dim*2, model_dim
                                        w2_qt,  # E, model_dim, inter_dim
                                        topk_weights, topk_ids,
                                        sorted_weights, sorted_ids,
                                        sorted_expert_ids, num_valid_ids,
                                        dtype=dtype,
                                        # [expert, inter_dim, 1]
                                        w2_scale=w2_scale,
                                        a2_scale=a2_scale,
                                        block_size=BLOCK_SIZE_M
                                        )

    # out_ref = torch_moe(input, w1, w2, topk_weights, topk_ids)

    # checkAllclose(out_ref, out2_ref, msg="[torch] 1_stage vs 2_stage")
    # w1b = torch.ones_like(w1b).cpu()*286331153
    # w1b = w1b.cuda()
    # print(sorted_ids)
    # print(sorted_expert_ids)
    # print(num_valid_ids)
    # # a1_scale = torch.full([w1.shape[1], E], a1_scale, device="cuda")
    out1_qt = ck_moe_stage1(a1_qt,
                                w1b,
                                w2b,
                                sorted_ids,
                                sorted_expert_ids,
                                num_valid_ids,
                                w1_scale, a1_scale,
                                dtype, topk, BLOCK_SIZE_M)
    print("ck out:", out1_qt)
    print("ref out:", out1_ref)
    checkAllclose(out1_ref, out1_qt,
                  msg=f'')
#
    # if use_g1u1:
    #    gate, up = out1_qt.split([inter_dim, inter_dim], dim=-1)
    #    input2 = F.silu(gate) * up
    # else:
    #    input2 = F.gelu(out1_qt)
    # # a2_qt, a2_scale = aiter.per_tensor_quant_fp8_hip(input2)
    # a2_qt, a2_scale = aiter.per_tensor_quant(input2,  quant_dtype=quant_dtype)

    # out2_qt, us = ck_moe_stage2(a2_qt,
    #                            w1_qt,
    #                            shuffle_weight222(w2_qt, layout=(32, 32)),
    #                            sorted_ids,
    #                            sorted_expert_ids,
    #                            sorted_weights,
    #                            num_valid_ids,
    #                            w2_scale, a2_scale,
    #                            dtype, topk, BLOCK_SIZE_M)
    # checkAllclose(out2_ref, out2_qt,
    #              msg=f'ck_moe_stage2:{us:.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:.2f} tflops......(quant:{quant_dtype})')

    # out_ck_qt, us = ck_moe_fused_2stages(input,
    #                                     shuffle_weight(
    #                                         w1_qt, layout=(32, 32)),
    #                                     shuffle_weight(
    #                                         w2_qt, layout=(32, 32)),
    #                                     topk_weights, topk_ids,
    #                                     w1_scale, w2_scale,
    #                                     #  block_size=BLOCK_SIZE_M
    #                                     )

    # checkAllclose(out2_ref, out_ck_qt,
    #              msg=f'ck_moe_fused_2stages:{us:.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:.2f} tflops......(quant:{quant_dtype})')

    #out_ck_nqt, us = ck_moe_fused_2stages(input,
    #                                      shuffle_weight(w1, layout=(32, 32)),
    #                                      shuffle_weight(w2, layout=(32, 32)),
    #                                      topk_weights, topk_ids,
    #                                      None, None,
    #                                      #   block_size=BLOCK_SIZE_M
    #                                      )
#
    #checkAllclose(out_ref, out_ck_nqt,
    #              msg=f'ck_moe_fused_2stages:{us:.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:.2f} tflops......(No quant)')
#

for dtype in [torch.float16]:
    for m in [32]:
        for dim in [1024]:
            for inter_dim in [1024]:
                expert, topk = 8, 2
                test_fmoe(dtype, m, dim, inter_dim, expert, topk,
                          quant='fp8quant', use_g1u1=True)