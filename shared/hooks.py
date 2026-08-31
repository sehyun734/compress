import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel

from shared.modules import Inps, fwd_blk


class StopForward(Exception):
    pass


@torch.no_grad()
def catch_blk_inps(model: PreTrainedModel, blk: nn.Module, calib: list[Tensor]) -> Inps:
    device = next(model.parameters()).device
    inps: Inps = []

    def hook(mod: nn.Module, a: tuple[Tensor, ...], kw: dict) -> None:
        inps.append((a, kw))
        raise StopForward

    handle = blk.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for tok_ids in calib:
            try:
                model(tok_ids.to(device))
            except StopForward:
                pass
    finally:
        handle.remove()
    return inps


@torch.no_grad()
def catch_lin_acts(blk: nn.Module, lin: nn.Linear, inps: Inps) -> Tensor:
    acts: list[Tensor] = []

    def hook(mod: nn.Linear, a: tuple[Tensor, ...]) -> None:
        acts.append(a[0])

    handle = lin.register_forward_pre_hook(hook)
    try:
        fwd_blk(blk, inps)
    finally:
        handle.remove()
    return torch.cat(acts)
