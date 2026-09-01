from dataclasses import dataclass, field

from tqdm import tqdm
from simple_parsing import parse
from torch import Tensor, nn
import torch
from transformers import PreTrainedModel

from shared.metrics import ppl
from shared.hooks import catch_blk_inps
from shared.load import load_calib, load_eval, load_llm
from shared.modules import Inps, fwd_blk, get_blks, get_lins
from shared.quant import apply_qparams, calc_qparams


@dataclass
class Args:
    model: str = "meta-llama/Llama-3.2-3B"
    bit: int = 4
    grp_size: int = 128
    col_blk_size: int = 128
    n_calib: int = 128
    seq_len: int = 2048
    r_damp: float = 0.01
    skip_lins: list[str] = field(
        default_factory=lambda: ["model.layers.1.mlp.down_proj"]
    )
    seed: int = 42


def calc_hess(blk: nn.Module, lin: nn.Linear, inps: Inps) -> Tensor:
    d_in = lin.in_features
    H = torch.zeros(d_in, d_in, device=lin.weight.device, dtype=torch.float32)
    n_tok = 0

    def hook(mod: nn.Module, a: tuple[Tensor, ...]) -> None:
        nonlocal H, n_tok
        x = a[0].reshape(-1, a[0].shape[-1]).float()
        H += x.mT @ x
        n_tok += x.shape[0]

    handle = lin.register_forward_pre_hook(hook)
    try:
        fwd_blk(blk, inps)
    finally:
        handle.remove()
    return H * (2 / n_tok)


def quant_weight(w: Tensor, H: Tensor, args: Args) -> Tensor:
    w = w.float().clone()
    d_out, d_in = w.shape

    i_dead = H.diagonal() == 0
    w[:, i_dead] = 0
    H.diagonal()[i_dead] = 1
    # H가 calib 기반이라 작은 고윳값 방향으로 과적합될수 있어서 damp하여 방지
    H.diagonal().add_(args.r_damp * H.diagonal().mean())  # 대각평균 = 고윳값평균
    H_inv: Tensor = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(H)),
        upper=True,
    )

    w_hat = torch.zeros_like(w)
    err = torch.zeros(d_out, args.col_blk_size, device=w.device)
    s: Tensor | None = None
    z: Tensor | None = None
    for i_start in range(0, d_in, args.col_blk_size):
        i_end = min(i_start + args.col_blk_size, d_in)
        for i_col in range(i_start, i_end):
            if i_col % args.grp_size == 0:
                s, z = calc_qparams(w[:, i_col : i_col + args.grp_size], args.bit)
                s = s.squeeze(-1)
                z = z.squeeze(-1)

            w_col = w[:, i_col]
            w_hat_col = apply_qparams(w_col, s, z, args.bit)
            err_col = (w_col - w_hat_col) / H_inv[i_col, i_col]

            w_hat[:, i_col] = w_hat_col
            err[:, i_col - i_start] = err_col
            w[:, i_col:i_end] -= torch.outer(err_col, H_inv[i_col, i_col:i_end])

        w[:, i_end:] -= err[:, : i_end - i_start] @ H_inv[i_start:i_end, i_end:]
    return w_hat


@torch.no_grad()
def gptq(model: PreTrainedModel, calib: list[Tensor], args: Args) -> None:
    """
    NOTE: Llama-3.2-3B에서 ppl 붕괴.
          logs/Llama-3.2-3B_probe_act에서 model.layers.1.mlp.down_proj을 보면
          peak 824, chan max 1.8로 소수 요소만 폭발. (아마 attn sink위한 massive act)
          이 입력에서 down_proj양자화 오차 엄청나져서 H까지 폭발. 그래서 gptq 보상도 폭발.
          무튼 1.mlp.down_proj만 skip하면 훨씬 좋아짐.
    """

    for blk_name, blk in tqdm(get_blks(model).items()):
        inps = catch_blk_inps(model, blk, calib)
        for lin_name, lin in get_lins(blk, blk_name).items():
            if lin_name in args.skip_lins:
                continue
            H = calc_hess(blk, lin, inps)
            w_hat = quant_weight(lin.weight.data, H, args)
            lin.weight.data = w_hat.to(lin.weight)


def main() -> None:
    args = parse(Args)
    print(args)
    model, tokenizer = load_llm(args.model)
    calib = load_calib(tokenizer, args.n_calib, args.seq_len, args.seed)
    gptq(model, calib, args)
    print(f"ppl={ppl(model, load_eval(tokenizer, args.seq_len)):.4f}")


if __name__ == "__main__":
    main()
