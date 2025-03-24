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
from aiter.fused_moe_bf16_asm import asm_moe, torch_moe, moe_sorting_ck, ck_moe_2stages, get_block_size
from aiter.ops.shuffle import shuffle_weight
from aiter import ActivationType

@perftest(num_iters=2, num_warmup=1)
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
    E=w1.shape[0]
    topk = topk_weight.shape[1]
    N = w1.shape[1]
    num_experts, model_dim, inter_dim = w2.shape
    # hidden_states = hidden_states.view(
    #     B, -1, D).repeat(1, topk, 1)

    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk

    # gose to quant D_w8a8/w8a8
    if fc1_scale is not None:
        w1 = (w1.view(-1, D) * fc1_scale.view(-1, 1)).view(num_experts, -1, D)
    if a1_scale is not None and w1_scale is not None:
        hidden_states = hidden_states * a1_scale
        w1 = w1 * w1_scale.view(E, -1, 1)

    hidden_states = hidden_states.view(
        B, -1, D).repeat(1, topk, 1)
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


@perftest(num_iters=2, num_warmup=1)
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

    E = w1.shape[0]
    token_num, topk = topk_ids.shape
    # M, _ = hidden_states.shape
    num_experts, model_dim, inter_dim = w2.shape
    max_num_m_blocks = sorted_expert_ids.shape[0]

    # gose to quant D_w8a8/w8a8
    if a2_scale is not None and w2_scale is not None:
        hidden_states = hidden_states * a2_scale
        w2 = w2 * w2_scale.view(E, -1, 1)

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
        (token_num, topk, D // 2),
        dtype=dtype,
        device=hidden_states.device,
    )
    aiter.ck_moe_stage1(hidden_states, w1, w2, sorted_token_ids,
                        sorted_expert_ids, num_valid_ids, out, topk, w1_scale, a1_scale, block_size)
                        
    return out
    # gate, up = out.split([inter_dim, inter_dim], dim=-1)
    # return F.silu(gate) * up


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


@perftest()
def ck_moe_fused_2stages(hidden_states,
                         # [expert(local_expert:EP), inter_dim(*2), dim] N,K
                         w1,
                         w2,  # [expert(local_expert:EP), dim, inter_dim]
                         topk_weight, topk_ids,
                         # following for int8 quant
                         # [expert(local_expert:EP), inter_dim, 1]
                         fc1_scale=None,
                         # [expert(local_expert:EP), model_dim, 1]
                         fc2_scale=None,
                         block_size=32,
                         a1_scale=None,
                         activation=ActivationType.Silu
                         ):
    return ck_moe_2stages(hidden_states, w1, w2, topk_weight, topk_ids,
                          fc1_scale, fc2_scale, block_size=block_size, a1_scale=a1_scale, activation=activation)


def test_fmoe(dtype, token, model_dim, inter_dim, E, topk, quant='No', use_g1u1=False, shared_E=0, activation=ActivationType.Silu):
    input = torch.rand((token, model_dim), dtype=dtype, device="cuda")  / 100.0
    if use_g1u1:
        w1 = torch.rand((E+shared_E, inter_dim*2, model_dim),
                         dtype=dtype, device="cuda")  / 100.0
    else:
        w1 = torch.rand((E+shared_E, inter_dim, model_dim),
                         dtype=dtype, device="cuda")
    w2 = torch.rand((E+shared_E, model_dim, inter_dim),
                     dtype=dtype, device="cuda")
    score = torch.rand((token, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    E, model_dim, inter_dim = w2.shape
    M, topk = topk_ids.shape
    BLOCK_SIZE_M = 128 #get_block_size(M, topk, E)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting_ck(topk_ids, topk_weights, E,
                                                                                           model_dim, dtype, BLOCK_SIZE_M)

    quant_dtype = torch.float8_e4m3fnuz
    if "perTensorQuant" in quant:
        w1_qt, w1_scale = aiter.pertoken_quant(w1.view(E, -1), quant_dtype=quant_dtype)
        w2_qt, w2_scale = aiter.pertoken_quant(w2.view(E, -1), quant_dtype=quant_dtype)
    else:
        w1_qt, w1_scale = aiter.pertoken_quant(w1, quant_dtype=quant_dtype)
        w2_qt, w2_scale = aiter.pertoken_quant(w2, quant_dtype=quant_dtype)
    
    w1_qt = w1_qt.view(w1.shape)
    w2_qt = w2_qt.view(w2.shape)
    # w1_scale = torch.ones_like(w1_scale)
    # w1_qt = torch.ones_like(w1_qt)
    
    if "perTensorQuant" in quant:
        a1_qt, a1_scale = aiter.per_tensor_quant(input,  quant_dtype=quant_dtype)
    else:
        a1_qt, a1_scale = aiter.pertoken_quant(input,  quant_dtype=quant_dtype)

    a1_scale = torch.ones_like(a1_scale) / 100
    # a1_qt = torch.ones_like(a1_qt)
    out1_ref, us_ref = torch_moe_stage1(a1_qt, w1_qt,
                                        w2_qt,
                                        topk_weights, topk_ids,
                                        dtype=dtype,
                                        fc1_scale=None,
                                        w1_scale=w1_scale,
                                        a1_scale=a1_scale,
                                        block_size=BLOCK_SIZE_M)
    if use_g1u1:
        gate, up = out1_ref.split([inter_dim, inter_dim], dim=-1)
        gate =F.silu(gate)
        print(gate, up)
        input2 = gate * up
    else:
        if activation == ActivationType.Silu:
            input2 = F.silu(out1_ref)
        else:
            input2 = F.gelu(out1_ref)
    # if "perTensorQuant" in quant:
    #     a2_qt, a2_scale = aiter.per_tensor_quant(input2,  quant_dtype=quant_dtype)
    # else:
    #     a2_qt, a2_scale = aiter.pertoken_quant(input2.view(M, -1),  quant_dtype=quant_dtype)
    #     # a2_qt, a2_scale = aiter.per_token_dynamic_quant_fp8_hip(input2.view(M, -1))
    #     a2_qt = a2_qt.view(M, topk, -1)
    # print(a2_qt.shape, a2_scale.shape)
    # out2_ref, us_ref = torch_moe_stage2(a2_qt,
    #                                     w1_qt,  # E, inter_dim*2, model_dim
    #                                     w2_qt,  # E, model_dim, inter_dim
    #                                     topk_weights, topk_ids,
    #                                     sorted_weights, sorted_ids,
    #                                     sorted_expert_ids, num_valid_ids,
    #                                     dtype=dtype,
    #                                     # [expert, inter_dim, 1]
    #                                     w2_scale=w2_scale,
    #                                     a2_scale= a2_scale if "perTensorQuant" in quant else a2_scale.view(M, -1, 1).repeat(1, topk, 1),
    #                                     block_size=BLOCK_SIZE_M
    #                                     )

    # out_ref = torch_moe(input, w1, w2, topk_weights, topk_ids, activation=activation)

    # checkAllclose(out_ref, out2_ref, msg="[torch] 1_stage vs 2_stage")
    us = 1
    out1_qt= ck_moe_stage1(a1_qt,
                                shuffle_weight(w1_qt, layout=(32, 32)),
                                w2,
                                sorted_ids,
                                sorted_expert_ids,
                                num_valid_ids,
                                w1_scale, a1_scale,
                                dtype, topk, BLOCK_SIZE_M)
    checkAllclose(input2, out1_qt,
    msg=f'{token} ck_moe_stage1:{us:.2f} us, {token*model_dim*inter_dim*2*topk*2/us/1000/1000:.2f} tflops......(quant:{quant_dtype})')
    print(input2)
    print(out1_qt)
    # if use_g1u1:
    #     gate, up = out1_qt.split([inter_dim, inter_dim], dim=-1)
    #     if activation == ActivationType.Silu:
    #         input2 = F.silu(gate) * up
    #     else:
    #         input2 = F.gelu(gate) * up
    # else:
    #     if activation == ActivationType.Silu:
    #         input2 = F.silu(out1_qt)
    #     else:
    #         input2 = F.gelu(out1_qt)
    # if "perTensorQuant" in quant:
    #     a2_qt, a2_scale = aiter.per_tensor_quant(input2,  quant_dtype=quant_dtype)
    # else:
    #     a2_qt, a2_scale = aiter.pertoken_quant(input2.view(M, -1),  quant_dtype=quant_dtype)
    #     # a2_qt, a2_scale = aiter.per_token_dynamic_quant_fp8_hip(input2.view(M, -1))
    #     a2_qt = a2_qt.view(M, topk, -1)

    # out2_qt, us = ck_moe_stage2(a2_qt,
    #                             w1_qt,
    #                             shuffle_weight(w2_qt, layout=(32, 32)),
    #                             sorted_ids,
    #                             sorted_expert_ids,
    #                             sorted_weights,
    #                             num_valid_ids,
    #                             w2_scale, a2_scale,
    #                             dtype, topk, BLOCK_SIZE_M)
    # checkAllclose(out2_ref, out2_qt,
    #               msg=f'{token} ck_moe_stage2:{us:.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:.2f} tflops......(quant:{quant_dtype})')

    # out_ck_qt, us = ck_moe_fused_2stages(input,
    #                                      shuffle_weight(
    #                                          w1_qt, layout=(32, 32)),
    #                                      shuffle_weight(
    #                                          w2_qt, layout=(32, 32)),
    #                                      topk_weights, topk_ids,
    #                                      w1_scale, w2_scale,
    #                                      activation=activation
    #                                      #  block_size=BLOCK_SIZE_M
    #                                      )
    # print(f'{token} ck_moe_fused_2stages:{us:.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:.2f} tflops......(quant:{quant_dtype})')
    # checkAllclose(out2_ref, out_ck_qt,
    #               msg=)

    # out_ck_nqt, us = ck_moe_fused_2stages(input,
    #                                       shuffle_weight(w1, layout=(32, 32)),
    #                                       shuffle_weight(w2, layout=(32, 32)),
    #                                       topk_weights, topk_ids,
    #                                       None, None,
    #                                      activation=activation
    #                                       #   block_size=BLOCK_SIZE_M
    #                                       )

    # checkAllclose(out_ref, out_ck_nqt,
    #               msg=f'ck_moe_fused_2stages:{us:.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:.2f} tflops......(No quant)')


for dtype in [torch.float16]:
    for m in [132]:
        for dim in [512]:
            for inter_dim in [4096]:
                expert, topk = 1, 1
                test_fmoe(dtype, m, dim, inter_dim, expert, topk,
                          quant='fp8perTokenQuant', use_g1u1=True, activation=ActivationType.Silu)
