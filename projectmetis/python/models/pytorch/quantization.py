from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch


def quantize_tensor(
    tensor: torch.Tensor,
    nbits: int = 4,
    block_size: int = 64,
    residual: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    tuple[int, ...],
    torch.Tensor | None,
]:
    """Quantize a tensor into block-wise symmetric uniform integers with sub-byte bit packing (uint8)."""
    orig_shape = tensor.shape
    orig_device = tensor.device
    orig_dtype = tensor.dtype

    work = tensor.detach().clone().to(torch.float32)
    if residual is not None and residual.shape == orig_shape:
        work = work + residual.to(device=orig_device, dtype=torch.float32)

    flat = work.reshape(-1)
    numel = flat.numel()

    if numel == 0:
        empty_packed = torch.empty((0,), dtype=torch.uint8, device=orig_device)
        empty_scales = torch.empty((0,), dtype=torch.float32, device=orig_device)
        empty_res = torch.zeros(orig_shape, dtype=orig_dtype, device=orig_device)
        return empty_packed, empty_scales, None, orig_shape, empty_res

    pad_len = (block_size - (numel % block_size)) % block_size
    if pad_len > 0:
        padded = torch.nn.functional.pad(flat, (0, pad_len), mode="constant", value=0.0)
    else:
        padded = flat

    blocks = padded.view(-1, block_size)
    qmax = (1 << (nbits - 1)) - 1  # 7 for 4-bit
    qmin = -(1 << (nbits - 1))  # -8 for 4-bit

    max_val = torch.max(torch.abs(blocks), dim=1, keepdim=True)[0]
    scales = torch.clamp(max_val / float(qmax), min=1e-8)

    scaled_blocks = blocks / scales
    quantized_blocks = torch.clamp(torch.round(scaled_blocks), qmin, qmax)

    # Compute reconstructed values for error-feedback residual
    dequantized_padded = (quantized_blocks * scales).view(-1)
    dequantized_flat = dequantized_padded[:numel]
    new_residual = (flat - dequantized_flat).view(orig_shape).to(orig_dtype)

    # Unsigned representation [0, 2^nbits - 1]
    unsigned_blocks = (quantized_blocks - qmin).to(torch.uint8)
    unsigned_flat = unsigned_blocks.view(-1)

    if nbits == 4:
        even_nibbles = unsigned_flat[0::2]
        odd_nibbles = unsigned_flat[1::2]
        packed = (even_nibbles & 0x0F) | ((odd_nibbles & 0x0F) << 4)
    elif nbits == 8:
        packed = unsigned_flat
    else:
        raise ValueError(f"Unsupported nbits: {nbits}. Supported: 4, 8.")

    return packed, scales.squeeze(-1), None, orig_shape, new_residual


def dequantize_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None = None,
    orig_shape: tuple[int, ...] | None = None,
    nbits: int = 4,
    block_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize packed integers back into floating-point representation."""
    if packed.numel() == 0:
        if orig_shape is not None:
            return torch.zeros(orig_shape, dtype=dtype, device=packed.device)
        return torch.empty((0,), dtype=dtype, device=packed.device)

    qmin = -(1 << (nbits - 1))

    if nbits == 4:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        unsigned = torch.stack([low, high], dim=-1).view(-1)
    elif nbits == 8:
        unsigned = packed
    else:
        raise ValueError(f"Unsupported nbits: {nbits}. Supported: 4, 8.")

    signed = unsigned.to(torch.float32) + float(qmin)
    blocks = signed.view(-1, block_size)

    scale_view = scales.view(-1, 1)
    recon_blocks = blocks * scale_view
    recon_flat = recon_blocks.view(-1)

    if orig_shape is not None:
        numel = 1
        for dim in orig_shape:
            numel *= dim
        recon_flat = recon_flat[:numel]
        return recon_flat.view(orig_shape).to(dtype)

    return recon_flat.to(dtype)


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    nbits: int = 4,
    block_size: int = 64,
    residuals: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, float]]:
    """Quantize PyTorch model state dict for MetisFL communication."""
    quantized_dict: dict[str, Any] = {
        "nbits": nbits,
        "block_size": block_size,
        "layers": {},
    }
    new_residuals: dict[str, torch.Tensor] = {}

    total_orig_bytes = 0
    total_quant_bytes = 0

    if residuals is None:
        residuals = {}

    for name, param in state_dict.items():
        res = residuals.get(name, None)
        orig_bytes = param.numel() * param.element_size()
        total_orig_bytes += orig_bytes

        if not param.is_floating_point():
            quantized_dict["layers"][name] = {
                "type": "passthrough",
                "tensor": param.detach().clone(),
            }
            total_quant_bytes += orig_bytes
            continue

        packed, scales, zeros, orig_shape, new_res = quantize_tensor(
            param, nbits=nbits, block_size=block_size, residual=res
        )

        quant_bytes = (
            packed.numel() * packed.element_size()
            + scales.numel() * scales.element_size()
        )
        total_quant_bytes += quant_bytes

        quantized_dict["layers"][name] = {
            "type": "quantized",
            "packed": packed,
            "scales": scales,
            "zeros": zeros,
            "orig_shape": orig_shape,
            "dtype": str(param.dtype),
        }
        new_residuals[name] = new_res

    compression_ratio = float(total_orig_bytes) / max(total_quant_bytes, 1)
    stats = {
        "orig_bytes": total_orig_bytes,
        "quant_bytes": total_quant_bytes,
        "compression_ratio": compression_ratio,
        "bandwidth_savings_pct": (1.0 - (total_quant_bytes / max(total_orig_bytes, 1)))
        * 100.0,
    }

    return quantized_dict, new_residuals, stats


def dequantize_state_dict(
    quantized_dict: dict[str, Any],
    device: torch.device | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Dequantize compressed payload back into state dict."""
    nbits = quantized_dict.get("nbits", 4)
    block_size = quantized_dict.get("block_size", 64)
    layers = quantized_dict.get("layers", {})

    dequantized_state: OrderedDict[str, torch.Tensor] = OrderedDict()

    for name, data in layers.items():
        if data.get("type") == "passthrough":
            tensor = data["tensor"]
            if device is not None:
                tensor = tensor.to(device)
            dequantized_state[name] = tensor
        else:
            packed = data["packed"]
            scales = data["scales"]
            zeros = data.get("zeros")
            orig_shape = data["orig_shape"]

            recon = dequantize_tensor(
                packed=packed,
                scales=scales,
                zeros=zeros,
                orig_shape=orig_shape,
                nbits=nbits,
                block_size=block_size,
            )
            if device is not None:
                recon = recon.to(device)
            dequantized_state[name] = recon

    return dequantized_state
