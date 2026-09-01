from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_parsing import parse
from torch import Tensor
from torch.optim import AdamW
from transformers import PreTrainedModel, get_cosine_with_min_lr_schedule_with_warmup
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from shared.metrics import ppl
from shared.hooks import catch_blk_inps, catch_lin_acts
from shared.load import load_calib, load_eval, load_llm, load_sft
from shared.modules import apply_rope, fwd_qkv, get_all_lins, get_blks, get_rope, sdpa
from shared.quant import mx_quant, mx_quant_ste
from shared.utils import print_args


@dataclass
class Args:
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # R1-Qwen-14B
    seq_len: int = 2048  # 25000
    n_sft: int = 8192  # 350_000_000
    n_taq: int = 2048  # 70_000_000
    n_calib: int = 64  # 256
    n_accum: int = 2
    lr_max: float = 1e-5
    r_lr_min: float = 0.01
    r_warmup: float = 0.03
    n_grid: int = 8
    grp_size: int = 16
    r_tau: float = 0.75
    lam: float = 0.1
    seed: int = 42


def sft(
    model: PreTrainedModel,
    samples: list[tuple[Tensor, Tensor]],
    args: Args,
    aux_loss: Callable[[], Tensor] | None = None,
) -> None:
    model.train()
    device = next(model.parameters()).device
    n_step = len(samples) // args.n_accum
    optim = AdamW(
        model.parameters(),
        lr=args.lr_max,
        weight_decay=0,  # scratch와 달리 ft에서 weight를 0으로 당기는 건 사전학습 정보를 지우는 것
    )
    sched = get_cosine_with_min_lr_schedule_with_warmup(
        optim,
        round(args.r_warmup * n_step),
        n_step,
        min_lr_rate=args.r_lr_min,
    )
    for i_step in range(n_step):
        for toks, labels in samples[
            i_step * args.n_accum : (i_step + 1) * args.n_accum
        ]:
            toks, labels = toks.to(device), labels.to(device)
            out = model(toks, labels=labels)
            loss: Tensor = out.loss
            if aux_loss is not None:
                loss = loss + aux_loss(out.logits, labels)
            (loss / args.n_accum).backward()
        optim.step()
        sched.step()
        optim.zero_grad()
    model.eval()


@torch.no_grad()
def search_scale(
    attn: nn.Module,
    acts: Tensor,
    cos: Tensor,
    sin: Tensor,
    args: Args,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    n_rep = attn.num_key_value_groups
    d_head = attn.head_dim

    q, k, v = fwd_qkv(attn, acts)  # (n_batch, n_head | n_kv_head, seq_len, d_head)
    q_rope, k_rope = apply_rope(q, k, cos, sin)
    v_hat = mx_quant(v, args.grp_size, 1, 2)
    y = sdpa(q_rope, k_rope, v, n_rep)

    k_max = k.abs().amax(dim=(0, 2))  # (n_kv_head, d_head)
    s_k = (
        torch.maximum(
            k_max[:, : d_head // 2],
            k_max[:, d_head // 2 :],
        )
        .clamp_min(eps)
        .repeat(1, 2)
    )
    s_q: Tensor = s_k.repeat_interleave(n_rep, dim=0)  # (n_head, d_head)
    k_mean = k_rope.mean(dim=(0, 2))  # (n_kv_head, d_head)

    err_best = float("inf")
    alpha_best = 0.0
    beta_best = 0.0
    for alpha in np.linspace(0.0, 1.0, args.n_grid, endpoint=False):
        q_rope, k_rope = apply_rope(
            q * (s_q**alpha).unsqueeze(1), k / (s_k**alpha).unsqueeze(1), cos, sin
        )
        for beta in np.linspace(0.0, 1.0, args.n_grid, endpoint=False):
            k_hat = mx_quant(k_rope - beta * k_mean.unsqueeze(1), args.grp_size, 1, 2)
            y_hat = sdpa(q_rope, k_hat, v_hat, n_rep)
            err = (y - y_hat).float().pow(2).mean().item()
            if err < err_best:
                err_best = err
                alpha_best = alpha
                beta_best = beta
    return s_q**alpha_best, s_k**alpha_best, beta_best * k_mean


def absorb_scale(attn: nn.Module, s_q: Tensor, s_k: Tensor) -> None:
    for lin, s in [(attn.q_proj, s_q), (attn.k_proj, 1 / s_k)]:
        s: Tensor = s.flatten().to(lin.weight)
        lin.weight.data.mul_(s.unsqueeze(-1))
        if lin.bias is not None:
            lin.bias.data.mul_(s)


def qfit(model: PreTrainedModel, calib: list[Tensor], args: Args) -> None:
    """
    (q * s) @ (k / s)^T

    kv cache만 양자화되고 q는 아니므로, k의 outlier를 q로 떠넘겨 k의 양자화 난이도를 낮춤.
    마치 AWQ. 똑같이 s가 다 흡수되어 추론 비용 없음.
    이때 rope가 (r, r+d/2)를 섞으므로 s를 쌍끼리 똑같이 관리.
    gqa에서는 head 개수가 다르므로 s_q, s_k로 shape만 분리해서 관리.

    shift는 softmax 덕에 자유로운데, k를 0 근처로 모아 quant 시 격자 활용도를 높임.
    blk마다 민감도가 달라 blk별로 잡음.

    그리고 kv cache는 x @ w와 달리 e1m2를 사용. s와 shift로 값이 이미 고르게 모여 있어
    범위가 넓은 e2m1보다 좁지만 촘촘한 e1m2가 더 나음.
    """
    for blk in get_blks(model).values():
        attn = blk.self_attn
        inps = catch_blk_inps(model, blk, calib)
        acts = catch_lin_acts(blk, attn.q_proj, inps)
        cos, sin = get_rope(inps)
        s_q, s_k, shift = search_scale(attn, acts, cos, sin, args)
        absorb_scale(attn, s_q, s_k)
        attn.register_buffer("shift", shift.unsqueeze(1).to(attn.k_proj.weight))


def patch(model: PreTrainedModel, args: Args) -> None:
    def fwd_lin(lin: nn.Linear, x: Tensor) -> Tensor:
        return F.linear(
            mx_quant_ste(x, args.grp_size, 2, 1),
            mx_quant_ste(lin.weight, args.grp_size, 2, 1),
            lin.bias,
        )

    def fwd_attn(
        attn: nn.Module,
        q: Tensor,
        k: Tensor,  # (n_batch, n_kv_head, seq_len, d_head)
        v: Tensor,
        mask: Tensor | None,
        **kw,
    ) -> tuple[Tensor, Tensor | None]:
        n_rep = attn.num_key_value_groups
        scale = kw.get("scaling", attn.scaling)
        k_hat = mx_quant_ste(k - attn.shift, args.grp_size, 1, 2)
        v_hat = mx_quant_ste(v, args.grp_size, 1, 2)
        y = sdpa(q, k_hat, v_hat, n_rep, mask, scale)
        return y.transpose(1, 2).contiguous(), None

    for lin in get_all_lins(model).values():
        lin.forward = MethodType(fwd_lin, lin)

    ALL_ATTENTION_FUNCTIONS["qfit"] = fwd_attn
    ALL_MASK_ATTENTION_FUNCTIONS.register("qfit", sdpa_mask)  # bool 또는 None 형태 mask
    model.config._attn_implementation = "qfit"


def calc_sem(logits: Tensor, labels: Tensor, args: Args, eps: float = 1e-8) -> Tensor:
    log_p = logits[:, :-1, :].float().log_softmax(dim=-1)
    h = -(log_p.exp() * log_p).sum(dim=-1)  # (n_batch, seq_len-1)
    h = h[labels[:, 1:] != -100]
    h_min = h.min()
    tau = h.quantile(args.r_tau)
    w = (1 - (h - h_min) / (tau - h_min + eps)).clamp_min(0)
    return args.lam * (w.detach() * h).mean()


def reqat(
    model: PreTrainedModel,
    calib: list[Tensor],
    samples: list[tuple[Tensor, Tensor]],
    args: Args,
) -> None:
    """
    숫자나 연산자는 low entropy, 접속어는 high entropy.
    이때 양자화가 low entropy 위치의 tail mass를 키워 이상한 토큰이 뽑히게 만듦. 이것이 문제.

    stage1에서 ft하고, stage2에서 같은 trace로 똑같이 ft(taq).
    왜? 미리 가중치가 문제를 풀어보게 해서 문제 이해 자체의 오차를 없애두기 위함.
    양자화 모델 오차 = 파라미터 품질 부족 + 양자화 오차인데, 앞쪽 오차를 ft로 미리 잡아두는 것.

    그리고 stage2에서는 말했듯 low entropy 위치일수록 h를 눌러 tail mass를 줄임.

    그런데 이때 막무가내로 하면 kv cache 품질이 별로라,
    qfit으로 kv cache 양자화 난이도를 낮춰 시작점을 잡음.
    """
    sft(model, samples, args)
    qfit(model, calib, args)
    patch(model, args)
    sft(model, samples[: args.n_taq], args, partial(calc_sem, args=args))


def main():
    args = parse(Args)
    print_args(args)
    model, tokenizer = load_llm(args.model)
    samples = load_sft(tokenizer, args.n_sft, args.seq_len, args.seed)
    calib = load_calib(tokenizer, args.n_calib, args.seq_len, args.seed)
    reqat(model, calib, samples, args)
    print(f"ppl={ppl(model, load_eval(tokenizer, args.seq_len)):.4f}")


if __name__ == "__main__":
    main()
