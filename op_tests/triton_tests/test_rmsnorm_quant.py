import torch
import pytest
import triton
import aiter
from aiter.ops.triton.rmsnorm import rms_norm
from aiter.ops.triton.rmsnorm import rmsnorm2d_fwd_with_add
from aiter.ops.triton.rmsnorm import rmsnorm2d_fwd_with_smoothquant
from aiter.ops.triton.rmsnorm import rmsnorm2d_fwd_with_dynamicquant


def torch_rmsnorm(x, g, out_dtype=torch.float16, epsilon=1e-6):
    M, N = x.shape
    # cast to float32 as the triton kernel
    x_f32 = x.float()
    g_f32 = g.float()
    rms = torch.sqrt(torch.sum(x_f32 * x_f32, dim=-1) * 1 / N)
    rsigma = 1.0 / rms
    rms_norm_f32 = x_f32 * rsigma.unsqueeze(1) * g_f32
    # rms_norm = rms_norm_f32.to(out_dtype)
    # return rms_norm
    return rms_norm_f32


def run_torch(input, weight, eps, residual=None, x_scale=None, y_scale_dtype=None):
    if residual is None:
        residual_out = None
        output = torch_rmsnorm(input, weight, input.dtype, eps)
    else:
        residual_out = input + residual
        output = torch_rmsnorm(residual_out, weight, residual_out.dtype, eps)
    if y_scale_dtype is None:
        y_scale = None
        output_q = output
    else:
        output_q, y_scale = aiter.pertoken_quant(output, x_scale=x_scale)
    return output_q, residual_out, y_scale, output


def run_triton(input, weight, eps, residual=None, x_scale=None, y_scale_dtype=None):
    out_before_quant = None
    if y_scale_dtype is None:
        y_scale = None
        if residual is None:
            residual_out = None
            output = rms_norm(input, weight, eps)
        else:
            residual_out = torch.empty_like(input)
            output = torch.empty_like(input)
            rmsnorm2d_fwd_with_add(output, input, residual, residual_out, weight, eps)
    elif x_scale is None:
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
        output = torch.empty(input.shape, dtype=torch.int8, device="cuda")
        if residual is None:
            residual_out = None
            rmsnorm2d_fwd_with_dynamicquant(output, input, y_scale, weight, eps)
        elif residual is not None:
            residual_out = torch.empty_like(input)
            aiter.rmsnorm2d_fwd_with_add_dynamicquant(
                output, input, residual, residual_out, y_scale, weight, eps
            )
    else:
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
        output = torch.empty(input.shape, dtype=torch.int8, device="cuda")
        if residual is None:
            residual_out = None
            rmsnorm2d_fwd_with_smoothquant(output, input, x_scale, y_scale, weight, eps)
        else:
            residual_out = torch.empty_like(input)
            out_before_quant = torch.empty_like(input)
            rmsnorm2d_fwd_with_add_smoothquant(
                output,
                input,
                residual,
                residual_out,
                x_scale,
                y_scale,
                weight,
                eps,
                out_before_quant=out_before_quant,
            )
    return output, residual_out, y_scale, out_before_quant


@pytest.mark.parametrize("in_dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("scale_dtype_str", ["fp32"])
@pytest.mark.parametrize(
    "M, N",
    [
        (1, 4),
        (2, 10),
        (8192, 4096),
        (4096, 8192),
        (8000, 8000),
        (1, 31744),
        (3, 65536),
        (1, 131072),
        (873, 1245),
        (23153, 45),
        (89999, 234),
    ],
)
def test_rmsnorm_smoothquant(M, N, in_dtype_str, scale_dtype_str):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    in_dtype = arg_to_torch_dtype[in_dtype_str]
    scale_dtype = arg_to_torch_dtype[scale_dtype_str]

    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=in_dtype)
    weight = torch.randn(N, device="cuda", dtype=in_dtype)
    x_scale = torch.randn(N, device="cuda", dtype=scale_dtype)

    (y_torch, _, yscale_torch, *_) = run_torch(
        x, weight, 1e-5, x_scale=x_scale, y_scale_dtype=scale_dtype
    )
    (y_triton, _, yscale_triton, *_) = run_triton(
        x, weight, 1e-5, x_scale=x_scale, y_scale_dtype=scale_dtype
    )

    triton.testing.assert_close(y_triton, y_torch, atol=1, rtol=0)
    triton.testing.assert_close(yscale_triton, yscale_torch, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("in_dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("scale_dtype_str", ["fp32"])
@pytest.mark.parametrize(
    "M, N",
    [
        (1, 4),
        (2, 10),
        (8192, 4096),
        (4096, 8192),
        (8000, 8000),
        (1, 31744),
        (3, 65536),
        (1, 131072),
        (873, 1245),
        (23153, 45),
        (89999, 234),
    ],
)
def test_rmsnorm_dynamicquant(M, N, in_dtype_str, scale_dtype_str):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    in_dtype = arg_to_torch_dtype[in_dtype_str]
    scale_dtype = arg_to_torch_dtype[scale_dtype_str]

    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=in_dtype)
    weight = torch.randn(N, device="cuda", dtype=in_dtype)

    (y_torch, _, yscale_torch, *_) = run_torch(
        x, weight, 1e-5, y_scale_dtype=scale_dtype
    )
    (y_triton, _, yscale_triton, *_) = run_triton(
        x, weight, 1e-5, y_scale_dtype=scale_dtype
    )

    triton.testing.assert_close(y_triton, y_torch, atol=1, rtol=0)
    triton.testing.assert_close(yscale_triton, yscale_torch, atol=1e-3, rtol=1e-3)
