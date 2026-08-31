from typing import Literal

import torch
from torch import Tensor


def calc_qparams(
    x_grp: Tensor,
    bit: int,
    is_sym: bool = False,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    if is_sym:
        x_max = x_grp.abs().amax(-1, keepdim=True)
        x_min = -x_max
    else:
        x_min = x_grp.amin(-1, keepdim=True).clamp_max(0)
        x_max = x_grp.amax(-1, keepdim=True).clamp_min(0)
    s: Tensor = ((x_max - x_min) / (2**bit - 1)).clamp_min(eps)
    z: Tensor = torch.round(-x_min / s)
    return s, z


def apply_qparams(x: Tensor, s: Tensor, z: Tensor, bit: int) -> Tensor:
    x_q = (torch.round(x / s) + z).clamp(0, 2**bit - 1)
    return (x_q - z) * s


@torch.no_grad()
def quant(
    x: Tensor,
    bit: int,
    gran: Literal["tensor", "grp"] = "tensor",
    dim: int = -1,
    grp_size: int | None = None,
    is_sym: bool = False,
) -> Tensor:
    match gran:
        case "tensor":
            x_grp = x.float().reshape(1, -1)
        case "grp":
            assert grp_size is not None
            assert x.shape[dim] % grp_size == 0
            x_grp = x.movedim(dim, -1).float().contiguous()
            x_grp = x_grp.reshape(*x_grp.shape[:-1], -1, grp_size)
        case _:
            raise NotImplementedError(gran)

    s, z = calc_qparams(x_grp, bit, is_sym)
    x_hat = apply_qparams(x_grp, s, z, bit)

    match gran:
        case "tensor":
            x_hat = x_hat.reshape(x.shape)
        case "grp":
            x_hat = x_hat.reshape(*x_hat.shape[:-2], -1).movedim(-1, dim)
    return x_hat.to(x.dtype)


def make_grid(n_exp: int, n_man: int) -> Tensor:
    bias = 2 ** (n_exp - 1) - 1
    exp = torch.arange(2**n_exp).float()
    man = torch.arange(2**n_man).float() / 2**n_man
    subnor = man * 2.0 ** (1 - bias)
    nor = (1 + man).outer(torch.exp2(exp[1:] - bias)).T.flatten()
    return torch.cat([subnor, nor])


def round_grid(x: Tensor, grid: Tensor) -> Tensor:
    bound = (grid[1:] + grid[:-1]) / 2
    return grid[torch.bucketize(x, bound)]


@torch.no_grad()
def mx_quant(
    x: Tensor,
    grp_size: int,
    n_e_exp: int = 2,
    n_e_man: int = 1,
    n_s_exp: int = 4,
    n_s_man: int = 3,
    dim: int = -1,
    eps: float = 1e-8,
) -> Tensor:
    assert x.shape[dim] % grp_size == 0
    x_grp = x.movedim(dim, -1).float().contiguous()
    x_grp = x_grp.reshape(*x_grp.shape[:-1], -1, grp_size)

    grid_e = make_grid(n_e_exp, n_e_man).to(x_grp.device)
    grid_s = make_grid(n_s_exp, n_s_man).to(x_grp.device)
    s = round_grid(
        x_grp.abs().amax(-1, keepdim=True) / grid_e[-1],
        grid_s,
    ).clamp_min(eps)
    x_q = round_grid((x_grp / s).abs().clamp_max(grid_e[-1]), grid_e)
    x_hat = x_q * s * x_grp.sign()

    x_hat = x_hat.reshape(*x_hat.shape[:-2], -1).movedim(-1, dim)
    return x_hat.to(x.dtype)


def mx_quant_ste(
    x: Tensor,
    grp_size: int,
    n_e_exp: int = 2,
    n_e_man: int = 1,
    n_s_exp: int = 4,
    n_s_man: int = 3,
    dim: int = -1,
    eps: float = 1e-8,
) -> Tensor:
    return (
        x
        + (
            mx_quant(x, grp_size, n_e_exp, n_e_man, n_s_exp, n_s_man, dim, eps) - x
        ).detach()
    )
