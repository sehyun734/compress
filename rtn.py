from dataclasses import dataclass

from simple_parsing import parse
import torch
from transformers import PreTrainedModel

from shared.load import load_eval, load_llm
from shared.metrics import ppl
from shared.modules import get_all_lins
from shared.quant import quant


@dataclass
class Args:
    model: str = "meta-llama/Llama-3.2-3B"
    bit: int = 4
    grp_size: int = 128
    seq_len: int = 2048


@torch.no_grad()
def rtn(model: PreTrainedModel, args: Args) -> None:
    for lin in get_all_lins(model).values():
        lin.weight.data = quant(lin.weight.data, args.bit, "grp", -1, args.grp_size)


def main() -> None:
    args = parse(Args)
    print(args)
    model, tokenizer = load_llm(args.model)
    rtn(model, args)
    print(f"ppl={ppl(model, load_eval(tokenizer, args.seq_len)):.4f}")


if __name__ == "__main__":
    main()
