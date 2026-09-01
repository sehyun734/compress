from dataclasses import dataclass

import torch
from simple_parsing import parse
from torch import Tensor
from transformers import PreTrainedModel

from shared.hooks import catch_blk_inps, catch_lin_acts
from shared.load import load_calib, load_llm
from shared.modules import get_blks, get_lins


@dataclass
class Args:
    model: str = "meta-llama/Llama-3.2-3B"
    n_calib: int = 16
    seq_len: int = 512
    seed: int = 42


@torch.no_grad()
def probe_act(model: PreTrainedModel, calib: list[Tensor]) -> None:
    print(
        f"{'lin':36}"
        f"{'peak':>10}"
        f"{'chan max':>10}"
        f"{'chan p99':>10}"  # 상위 1%. max와 벌어지면 outlier가 소수
        f"{'chan med':>10}"
        f"{'chan max/med':>14}"  # 채널이 치우쳐진 정도. 크면 chan outlier
        f"{'chan kurt':>11}"  # 높으면 소수 chan만 outlier, 낮으면 outlier 광범위
        f"{'tok max':>10}"
        f"{'tok p99':>10}"
        f"{'tok med':>10}"
        f"{'tok max/med':>13}"  # 토큰이 치우쳐진 정도. 크면 attention sink
        f"{'tok kurt':>10}"
    )
    for blk_name, blk in get_blks(model).items():
        inps = catch_blk_inps(model, blk, calib)
        for lin_name, lin in get_lins(blk, blk_name).items():
            acts = catch_lin_acts(blk, lin, inps)
            x = acts.reshape(-1, acts.shape[-1]).abs().float()

            x_chan = x.mean(0)
            chan_max = x_chan.amax()
            chan_p99 = x_chan.quantile(0.99)
            chan_mid = x_chan.median()
            r_chan = float(chan_max / chan_mid)
            kurt_chan = (x_chan - x_chan.mean()).pow(4).mean() / x_chan.var().pow(2)

            x_tok = x.mean(1)
            tok_max = x_tok.amax()
            tok_p99 = x_tok.quantile(0.99)
            tok_mid = x_tok.median()
            r_tok = float(tok_max / tok_mid)
            kurt_tok = (x_tok - x_tok.mean()).pow(4).mean() / x_tok.var().pow(2)

            print(
                f"{lin_name:36}"
                f"{float(x.max()):>10.3f}"
                f"{float(chan_max):>10.3f}"
                f"{float(chan_p99):>10.3f}"
                f"{float(chan_mid):>10.3f}"
                f"{r_chan:>14.1f}"
                f"{float(kurt_chan):>11.1f}"
                f"{float(tok_max):>10.3f}"
                f"{float(tok_p99):>10.3f}"
                f"{float(tok_mid):>10.3f}"
                f"{r_tok:>13.1f}"
                f"{float(kurt_tok):>10.1f}"
            )


def main() -> None:
    args = parse(Args)
    print(args)
    model, tokenizer = load_llm(args.model)
    calib = load_calib(tokenizer, args.n_calib, args.seq_len, args.seed)
    probe_act(model, calib)


if __name__ == "__main__":
    main()
