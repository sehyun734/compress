from dataclasses import dataclass
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_parsing import parse
from torch import Tensor
from transformers import PreTrainedModel
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import repeat_kv

from shared.load import load_eval, load_llm
from shared.metrics import ppl
from shared.modules import get_all_lins
from shared.quant import make_grid, round_grid


@dataclass
class Args:
    model: str = "meta-llama/Llama-3.2-3B"
    grp_size: int = 32
    seq_len: int = 2048


def amx_quant(
    x: Tensor,
    grp_size: int,
    n_e_exp: int = 2,
    n_e_man: int = 1,
    n_s_exp: int = 5,
    n_s_man: int = 2,
    dim: int = -1,
    eps: float = 1e-8,
) -> Tensor:
    assert x.shape[dim] % grp_size == 0
    x_grp = x.movedim(dim, -1).float().contiguous()
    x_grp = x_grp.reshape(*x_grp.shape[:-1], -1, grp_size)

    grid_e = make_grid(n_e_exp, n_e_man).to(x_grp.device)
    grid_s = make_grid(n_s_exp, n_s_man).to(x_grp.device)
    x_max_pos = x_grp.clamp_min(0).amax(-1, keepdim=True)
    x_max_neg = x_grp.clamp_max(0).abs().amax(-1, keepdim=True)
    s_pos = round_grid(x_max_pos / grid_e[-1], grid_s).clamp_min(eps)
    s_neg = round_grid(x_max_neg / grid_e[-1], grid_s).clamp_min(eps)
    s = torch.where(x_grp >= 0, s_pos, s_neg)
    x_q = round_grid((x_grp / s).abs(), grid_e)
    x_hat = x_q * s * x_grp.sign()

    x_hat = x_hat.reshape(*x_hat.shape[:-2], -1).movedim(-1, dim)
    return x_hat.to(x.dtype)


def amxfp4(model: PreTrainedModel, args: Args) -> None:
    def fwd_lin(lin: nn.Linear, x: Tensor) -> Tensor:
        return F.linear(
            amx_quant(x, args.grp_size),
            amx_quant(lin.weight, args.grp_size),
            lin.bias,
        )

    def fwd_attn(
        attn: nn.Module,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None,
        **kw,
    ) -> tuple[Tensor, None]:
        n_rep = attn.num_key_value_groups
        scale = kw.get("scaling", attn.scaling)

        q_hat = amx_quant(q, args.grp_size)
        k_hat = repeat_kv(amx_quant(k, args.grp_size), n_rep)
        qk: Tensor = q_hat @ k_hat.transpose(-1, -2) * scale
        if mask is not None:
            qk = qk + mask  # eager mask 필요
        p = qk.softmax(-1, dtype=torch.float32).to(q.dtype)

        p_hat = amx_quant(p, args.grp_size)
        v_hat = repeat_kv(amx_quant(v, args.grp_size, dim=-2), n_rep)
        y = p_hat @ v_hat
        return y.transpose(1, 2).contiguous(), None

    for lin in get_all_lins(model).values():
        lin.forward = MethodType(fwd_lin, lin)

    ALL_ATTENTION_FUNCTIONS["amxfp4"] = fwd_attn
    ALL_MASK_ATTENTION_FUNCTIONS.register("amxfp4", eager_mask)  # 직접 더해야 하는 경우
    model.config._attn_implementation = "amxfp4"


def main() -> None:
    args = parse(Args)
    print(args)
    model, tokenizer = load_llm(args.model)
    amxfp4(model, args)
    print(f"ppl={ppl(model, load_eval(tokenizer, args.seq_len)):.4f}")


if __name__ == "__main__":
    main()
