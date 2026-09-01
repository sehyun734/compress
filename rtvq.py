from dataclasses import dataclass, field

import torch
from simple_parsing import parse
from transformers import PreTrainedModel

from shared.load import load_llm
from shared.quant import quant
from shared.utils import print_args


@dataclass
class Args:
    model: str = "meta-llama/Llama-3.2-1B"
    models_ft: list[str] = field(default_factory=lambda: [])
    lam: float = 0.4
    bit_base: int = 3
    bit_off: int = 2


@torch.no_grad()
def rtvq(model: PreTrainedModel, args: Args) -> None:
    """
    base = theta_ft_avg - theta_snap          # 공유, 한 번만 저장
    off  = theta_ft - theta_ft_avg            # task당
    tau  = base + off = theta_ft - theta_snap

    양자화 오차 상한: |err| <= (x_max - x_min) / (2 * (2**bit - 1))
    따라서 오차를 줄이려면 bit(분모)를 늘리거나 rng(분자)를 줄여야 함.

    TVQ는 theta_ft 대신 tau를 양자화하여 rng를 줄임.
    RTVQ는 더 나아가 tau를 base와 off로 쪼개고, 더 좁은 off를 양자화하여 rng를 한 번 더 줄임.
    또한 base는 공유되므로 저장 부담이 적어 더 많은 bit 할당 가능.
    """
    theta = dict(model.named_parameters())
    theta_snap = {name: param.data.clone() for name, param in theta.items()}

    theta_ft_avg = {}
    for model_ft, _ in map(load_llm, args.models_ft):
        for name, param_ft in model_ft.named_parameters():
            if name not in theta_ft_avg:
                theta_ft_avg[name] = param_ft.data.float().clone()
            else:
                theta_ft_avg[name].add_(param_ft.data)
        del model_ft
        torch.cuda.empty_cache()
    for name in theta_ft_avg:
        theta_ft_avg[name].div_(len(args.models_ft))

    base_hat = {}
    for name, w in theta_snap.items():
        w_ft_avg = theta_ft_avg[name]
        base = w_ft_avg - w
        base_hat[name] = quant(base, args.bit_base, "tensor").to(w.dtype)
    del theta_ft_avg

    for model_ft, _ in map(load_llm, args.models_ft):
        for name, param_ft in model_ft.named_parameters():
            w = theta_snap[name]
            w_ft_avg_hat = w + base_hat[name]
            w_ft = param_ft.data
            off = w_ft - w_ft_avg_hat
            off_hat = quant(off, args.bit_off, "tensor")
            tau_hat = off_hat + base_hat[name]
            theta[name].data.add_(tau_hat, alpha=args.lam)
        del model_ft
        torch.cuda.empty_cache()


def main():
    args = parse(Args)
    print_args(args)
    model, _ = load_llm(args.model)
    rtvq(model, args)
    # TODO: 평가


if __name__ == "__main__":
    main()
