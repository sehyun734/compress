from dataclasses import dataclass

import numpy as np
import torch
from simple_parsing import parse
from torch import Tensor, nn
from transformers import PreTrainedModel

from shared.metrics import ppl
from shared.hooks import catch_blk_inps, catch_lin_acts
from shared.load import load_calib, load_eval, load_llm
from shared.modules import Inps, fwd_mod, get_absorb_grps, get_blks, get_lins
from shared.quant import quant
from shared.utils import print_args


@dataclass
class Args:
    model: str = "meta-llama/Llama-3.2-3B"
    bit: int = 4
    grp_size: int = 128
    n_calib: int = 128
    seq_len: int = 2048
    n_grid: int = 20
    do_vo: bool = False
    seed: int = 42


@torch.no_grad()
def search_scale(
    mod_out: nn.Module,
    lins: dict[str, nn.Linear],
    acts: Tensor,  # (n_batch, seq_len, d_in)
    inps: Inps,
    args: Args,
    eps: float = 1e-4,
) -> Tensor:
    x_mean = acts.abs().flatten(0, -2).float().mean(0)  # (d_in,)
    y_fp = fwd_mod(mod_out, acts, inps)
    snap = {k: v.clone() for k, v in mod_out.state_dict().items()}

    err_best = float("inf")
    s_best = torch.ones_like(x_mean)
    for alpha in np.linspace(0.0, 1.0, args.n_grid, endpoint=False):
        s = (x_mean**alpha).clamp_min(eps)
        s = s / (s.amax() * s.amin()).sqrt()
        for lin in lins.values():
            s_w = s.to(lin.weight)
            w_hat = quant(lin.weight.data * s_w, args.bit, "grp", -1, args.grp_size)
            lin.weight.data = w_hat / s_w
        y_hat = fwd_mod(mod_out, acts, inps)
        err = (y_fp - y_hat).float().pow(2).mean().item()
        if err < err_best:
            err_best = err
            s_best = s
        mod_out.load_state_dict(snap)
    return s_best


def absorb(prev: nn.Module, lins: dict[str, nn.Linear], s: Tensor) -> None:
    s = s.to(prev.weight)
    if isinstance(prev, nn.Linear):
        prev.weight.data.div_(s.unsqueeze(-1))
    else:
        prev.weight.data.div_(s)
    if getattr(prev, "bias", None) is not None:
        prev.bias.data.div_(s)
    for lin in lins.values():
        lin.weight.data.mul_(s.to(lin.weight))


@torch.no_grad()
def search_scale_vo(
    attn: nn.Module,
    acts: Tensor,
    inps: Inps,
    args: Args,
    eps: float = 1e-4,
) -> Tensor:
    """
    v를 거치면 (n_kv_head, d_head)인데 o_proj 입력은 (n_head, d_head).
    그래서 gqa면 채널이 1:1로 대응하지 않아 s를 못 잡음. 기존 awq는 이를 그냥 건너뜀.

    이를 reqat가 s_k를 repeat 하던 것과 똑같이,
    kv head 단위로 s를 잡고 repeat_interleave로 펴서 흡수시킴.
    값은 같은데 shape만 다른 것.
    """
    n_rep = attn.num_key_value_groups
    o_proj = attn.o_proj
    x_mean = acts.abs().flatten(0, -2).float().mean(0)  # (d_in,)
    x_mean = x_mean.reshape(-1, n_rep, attn.head_dim).mean(1)  # (n_kv_head, d_head)
    y_fp = fwd_mod(o_proj, acts, inps)
    w_snap = o_proj.weight.data.clone()

    err_best = float("inf")
    s_best = torch.ones_like(x_mean)
    for alpha in np.linspace(0.0, 1.0, args.n_grid, endpoint=False):
        s = (x_mean**alpha).clamp_min(eps)
        s = s / (s.amax() * s.amin()).sqrt()
        s_rep = s.repeat_interleave(n_rep, dim=0).flatten().to(o_proj.weight)
        w_hat = quant(o_proj.weight.data * s_rep, args.bit, "grp", -1, args.grp_size)
        o_proj.weight.data = w_hat / s_rep
        y_hat = fwd_mod(o_proj, acts, inps)
        err = (y_fp - y_hat).float().pow(2).mean().item()
        o_proj.weight.data = w_snap
        if err < err_best:
            err_best = err
            s_best = s
    return s_best


def absorb_vo(attn: nn.Module, s: Tensor) -> None:
    s_v: Tensor = s.flatten().to(attn.v_proj.weight)
    s_o: Tensor = s.repeat_interleave(attn.num_key_value_groups, dim=0).flatten()
    attn.v_proj.weight.data.div_(s_v.unsqueeze(-1))
    if attn.v_proj.bias is not None:
        attn.v_proj.bias.data.div_(s_v)
    attn.o_proj.weight.data.mul_(s_o.to(attn.o_proj.weight))


@torch.no_grad()
def awq(model: PreTrainedModel, calib: list[Tensor], args: Args) -> None:
    """
    좋은 양자화는 weight만 볼 게 아니라 act도 같이 봐야 함.
    중요한 발견은 weight의 chan에 s를 곱해도 grp(d_in 방향)의 max가 잘 바뀌지 않는다는 것.
    그래서 반올림 오차 자체는 어느 정도까지는 그대로지만,
    양자화 후 s로 나눠 보상할 때 그 오차가 1/s로 축소됨.

    그렇다면 s를 어떻게 고르나?
    act가 큰 chan일수록 대응하는 weight의 양자화 오차가 증폭됨.
    그래서 mean(abs(x))로 chan 중요도를 잼.
    중요한 chan은 알아서 큰 s, 덜 중요한 chan은 알아서 작은 s가 됨.
    max를 안 건드릴 만큼만 키워야 하니 s = mean(abs(x))^alpha로 두고
    alpha를 [0, 1)에서 grid로 훑어 grp별 최적을 고름.

    또한 s는 prev에 1/s로, 대상 lin에 s로 흡수되어 추론 비용 없음.
    """
    for blk in get_blks(model).values():
        inps = catch_blk_inps(model, blk, calib)
        attn = blk.self_attn
        for prev, lins, mod_out in get_absorb_grps(model, blk):
            lin, *_ = lins.values()  # grp 내 lin들의 acts는 어차피 모두 같음
            acts = catch_lin_acts(blk, lin, inps)
            s = search_scale(mod_out, lins, acts, inps, args)
            absorb(prev, lins, s)

            for lin in lins.values():
                # 뒤 grp이 prev로 써야 하는 lin은 s가 또 곱해지니 지금은 넘어감
                if lin is blk.mlp.up_proj or (args.do_vo and lin is attn.v_proj):
                    continue
                lin.weight.data = quant(
                    lin.weight.data, args.bit, "grp", -1, args.grp_size
                )

            # o_proj은 어느 grp의 lins에도 없어 여기서 처리
            if mod_out is attn:
                if args.do_vo:
                    acts = catch_lin_acts(blk, attn.o_proj, inps)
                    s = search_scale_vo(attn, acts, inps, args)
                    absorb_vo(attn, s)
                    attn.v_proj.weight.data = quant(
                        attn.v_proj.weight.data, args.bit, "grp", -1, args.grp_size
                    )
                attn.o_proj.weight.data = quant(
                    attn.o_proj.weight.data, args.bit, "grp", -1, args.grp_size
                )

            if isinstance(prev, nn.Linear):
                prev.weight.data = quant(
                    prev.weight.data, args.bit, "grp", -1, args.grp_size
                )


def main() -> None:
    args = parse(Args)
    print_args(args)
    model, tokenizer = load_llm(args.model)
    calib = load_calib(tokenizer, args.n_calib, args.seq_len, args.seed)
    awq(model, calib, args)
    print(f"ppl={ppl(model, load_eval(tokenizer, args.seq_len)):.4f}")


if __name__ == "__main__":
    main()
