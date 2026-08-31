import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

type Inps = list[tuple[tuple[Tensor, ...], dict]]


def get_blks(model: PreTrainedModel) -> dict[str, nn.Module]:
    match model.config.model_type:
        case "llama" | "qwen2":
            blks = model.model.layers
        case _:
            raise NotImplementedError(model.config.model_type)
    blks_name = next(name for name, mod in model.named_modules() if mod is blks)
    return {f"{blks_name}.{i}": blk for i, blk in enumerate(blks)}


def get_lins(blk: nn.Module, blk_name: str = "") -> dict[str, nn.Linear]:
    return {
        f"{blk_name}.{lin_name}" if blk_name else lin_name: lin
        for lin_name, lin in blk.named_modules()
        if isinstance(lin, nn.Linear)
    }


def get_all_lins(model: PreTrainedModel) -> dict[str, nn.Linear]:
    return {
        lin_name: lin
        for blk_name, blk in get_blks(model).items()
        for lin_name, lin in get_lins(blk, blk_name).items()
    }


def get_rope(inps: Inps) -> tuple[Tensor, Tensor]:
    return inps[0][1]["position_embeddings"]


@torch.no_grad()
def fwd_blk(blk: nn.Module, inps: Inps) -> Tensor:
    ys = []
    for a, kw in inps:
        y = blk(*a, **kw)
        if isinstance(y, tuple):
            y = y[0]
        ys.append(y)
    return torch.cat(ys)


def fwd_mod(mod: nn.Module, acts: Tensor, inps: Inps) -> Tensor:
    if hasattr(mod, "q_proj"):
        return mod(acts, position_embeddings=get_rope(inps))[0]
    return mod(acts)


@torch.no_grad()
def fwd_qkv(
    attn: nn.Module,
    acts: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:  # (n_batch, n_head | n_kv_head, seq_len, d_head)
    d_head = attn.head_dim
    return (
        attn.q_proj(acts).float().unflatten(-1, (-1, d_head)).transpose(1, 2),
        attn.k_proj(acts).float().unflatten(-1, (-1, d_head)).transpose(1, 2),
        attn.v_proj(acts).float().unflatten(-1, (-1, d_head)).transpose(1, 2),
    )


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    return apply_rotary_pos_emb(q, k, cos, sin)


def sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    n_rep: int,
    mask: Tensor | None = None,
    scale: Tensor | None = None,
) -> Tensor:
    return F.scaled_dot_product_attention(
        q,
        repeat_kv(k, n_rep),
        repeat_kv(v, n_rep),
        attn_mask=mask,
        scale=scale,
        # sdpa_mask가 None인 경우는 prefill과 decode.
        # causal은 prefill에서만 필요 (decode는 query가 하나뿐이라 막을 게 없음)
        # decode는 반드시 1. seq_len > 1이 prefill
        is_causal=mask is None and q.shape[-2] > 1,  # 알아서 mask
    )


def get_absorb_grps(
    model: PreTrainedModel,
    blk: nn.Module,
) -> list[tuple[nn.Module, dict[str, nn.Linear], nn.Module]]:
    match model.config.model_type:
        case "llama" | "qwen2":
            attn = blk.self_attn
            mlp = blk.mlp
            return [
                (
                    blk.input_layernorm,
                    {
                        f"self_attn.{name}": getattr(attn, name)
                        for name in ("q_proj", "k_proj", "v_proj")
                    },
                    attn,
                ),
                (
                    blk.post_attention_layernorm,
                    {"mlp.gate_proj": mlp.gate_proj, "mlp.up_proj": mlp.up_proj},
                    mlp,
                ),
                (
                    mlp.up_proj,
                    {"mlp.down_proj": mlp.down_proj},
                    mlp.down_proj,
                ),
            ]
        case _:
            raise NotImplementedError(model.config.model_type)
