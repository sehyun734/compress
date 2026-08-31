from dataclasses import dataclass, field

from tqdm import tqdm
import torch
from simple_parsing import parse
from transformers import PreTrainedModel

from shared.load import load_llm
from shared.quant import quant


@dataclass
class Args:
    model: str = "meta-llama/Llama-3.2-1B"
    models_ft: list[str] = field(default_factory=lambda: [])
    lam: float = 0.4
    bit: int = 3


@torch.no_grad()
def tvq(model: PreTrainedModel, args: Args) -> None:
    """
    tau = theta_ft - theta_snap  # task당

    양자화 오차 상한: err <= (x_max - x_min) / (2 * (2**bit - 1))
    따라서 오차를 줄이려면 bit(분모)를 늘리거나 rng(분자)를 줄여야 함.

    tau는 theta_ft보다 rng가 약 10배(논문측정) 좁으므로, theta_ft 대신 tau 양자화.
    그러나 3bit까지만 유지되고 2bit에서 무너짐. 이를 개선한 것이 RTVQ.
    """
    theta = dict(model.named_parameters())
    theta_snap = {name: param.data.clone() for name, param in theta.items()}

    for model_ft, _ in tqdm(map(load_llm, args.models_ft), total=len(args.models_ft)):
        for name, param_ft in model_ft.named_parameters():
            w = theta_snap[name]
            w_ft = param_ft.data
            tau = w_ft - w
            tau_hat = quant(tau, args.bit, "tensor")
            theta[name].data.add_(tau_hat, alpha=args.lam)
        del model_ft
        torch.cuda.empty_cache()


def main():
    args = parse(Args)
    print(args)
    model, _ = load_llm(args.model)
    tvq(model, args)
    # TODO: 평가


if __name__ == "__main__":
    main()
