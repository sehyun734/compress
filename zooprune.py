import copy
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from simple_parsing import parse
from torch import Tensor
from transformers import PreTrainedModel

from shared.load import load_vlm
from shared.metrics import gqa, pope
from shared.vlm import get_img_id, get_lm, get_proj
from shared.utils import print_args


@dataclass
class Args:
    model: str = "llava-hf/llava-1.5-7b-hf"
    limit: int = 32
    n_keep: int = 64
    n_pert: int = 64
    mu: float = 0.01
    seed: int = 42


@torch.no_grad()
def zo_sens(
    x_vis: Tensor,
    proj: Callable[[Tensor], Tensor],
    n_pert: int,
    seed: int,
    mu: float,
    eps: float = 1e-8,
) -> Tensor:
    """
    수치해석으로 grad의 norm 근사하는 zeroth-order 기법.
    무작위 u(섭동) n_pert개 방향에서 잰 미분의 norm. 그 평균이 sens.
    """
    gen = torch.Generator(device=x_vis.device).manual_seed(seed)
    sens = torch.zeros(x_vis.shape[:-1], dtype=torch.float32, device=x_vis.device)
    for _ in range(n_pert):
        u = torch.randn(
            x_vis.shape[-1],
            generator=gen,
            dtype=x_vis.dtype,
            device=x_vis.device,
        )
        u_unit = u / u.norm().clamp_min(eps)
        delta = (proj(x_vis + mu * u_unit) - proj(x_vis - mu * u_unit)) / (2 * mu)
        sens.add_(delta.norm(dim=-1))
    sens.div_(n_pert)
    return sens


@torch.no_grad()
def pick_toks(
    sens: Tensor,
    x_lang: Tensor,
    n_keep: int,
    eps: float = 1e-8,
) -> list[int]:
    """
    sens와 diversity의 곱으로 매 스텝 가장 높은 토큰 하나씩 고름.
    직관은 sens만 쓰면 한 영역에 몰려 서로 중복된다는 것.
    """
    sens_min = sens.amin()
    sens_max = sens.amax()
    sens_hat = (sens - sens_min) / (sens_max - sens_min).clamp_min(eps)
    x_hat = x_lang / x_lang.norm(dim=-1, keepdim=True).clamp_min(eps)
    div = torch.ones_like(sens_hat)
    i_keep: list[int] = []
    for _ in range(min(n_keep, sens.shape[0])):
        if i_keep:
            div = 1 - (x_hat @ x_hat[i_keep].mT).amax(dim=1)
        score = sens_hat * div
        score[i_keep] = float("-inf")
        i_keep.append(int(score.argmax()))
    return i_keep


@torch.no_grad()
def zooprune(model: PreTrainedModel, args: Args) -> None:
    """
    img -> vis -> lang 흐름에서 x_img가 x_txt보다 압도적으로 많음.
    이것이 추론 비용의 주범이므로 x_img에서 중요한 것만 남기고 나머지를 prune.

    중요도 기준은 sens. 토큰을 흔들었을 때 출력이 얼마나 흔들리는가.
    sens만 쓰면 한 영역에 몰려 중복되므로 diversity를 곱해 함께 봄.
    기존 방법은 attn 기반이라 불안정하거나, diversity만 봐서 핵심 영역을 놓침.

    정공법은 encoder까지 거슬러 재야 하지만, 너무 비쌈.
    이때 보다 싼 proj 입출력으로 대신 재도 순위 상관이 유지되어 이를 활용(논문측정)
    """
    proj = get_proj(model)
    img_id = get_img_id(model)
    proj_snap = copy.deepcopy(proj).float()
    tok_ids: Tensor | None = None
    i_keep: list[int] | None = None

    def hook_ids(mod: nn.Module, a: tuple, kw: dict) -> None:
        nonlocal tok_ids
        tok_ids = kw.get("input_ids")

    def hook_proj(mod: nn.Module, a: tuple[Tensor, ...], out: Tensor) -> None:
        nonlocal i_keep
        x_vis = a[0].flatten(0, 1).float()
        x_lang = out.flatten(0, 1).float()
        sens = zo_sens(x_vis, proj_snap, args.n_pert, args.seed, args.mu)
        i_keep = pick_toks(sens, x_lang, args.n_keep)

    def hook_lm(mod: nn.Module, a: tuple, kw: dict) -> tuple[tuple, dict] | None:
        """
        hook_proj에서 x_lang을 prune해서 보내도 소용없음.
        proj 직후에 input_ids의 n_img_tok과 x_lang의 n_tok을 비교해 다르면 쳐내기 때문.
        그래서 hook_proj은 x_lang을 그대로 흘려보내고 i_keep만 기록하고,
        개수 검사를 통과한 여기서 i_keep 기준으로 자름.
        """
        nonlocal i_keep
        if i_keep is None or tok_ids is None:
            return None
        x_emb = kw["inputs_embeds"]
        assert x_emb.shape[0] == 1
        i_img = (tok_ids[0] == img_id).nonzero().flatten()
        i_txt = (tok_ids[0] != img_id).nonzero().flatten()
        i_keep_seq = torch.cat([i_txt, i_img[i_keep]]).sort().values
        kw["inputs_embeds"] = x_emb[:, i_keep_seq]
        kw["position_ids"] = i_keep_seq.unsqueeze(0)
        kw["cache_position"] = torch.arange(i_keep_seq.shape[0], device=x_emb.device)
        if kw.get("attention_mask") is not None:
            kw["attention_mask"] = kw["attention_mask"][:, i_keep_seq]
        i_keep = None
        return a, kw

    model.register_forward_pre_hook(hook_ids, with_kwargs=True)
    proj.register_forward_hook(hook_proj)
    get_lm(model).register_forward_pre_hook(hook_lm, with_kwargs=True)


def main():
    args = parse(Args)
    print_args(args)
    model, _ = load_vlm(args.model)
    zooprune(model, args)
    print(f"pope={pope(model, args.limit):.4f}")
    print(f"gqa={gqa(model, args.limit):.4f}")


if __name__ == "__main__":
    main()
