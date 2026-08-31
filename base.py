from dataclasses import dataclass

from simple_parsing import parse

from shared.load import load_eval, load_llm
from shared.metrics import ppl


@dataclass
class Args:
    model: str = "meta-llama/Llama-3.2-3B"
    seq_len: int = 2048


def main() -> None:
    args = parse(Args)
    print(args)
    model, tokenizer = load_llm(args.model)
    print(f"ppl={ppl(model, load_eval(tokenizer, args.seq_len)):.4f}")


if __name__ == "__main__":
    main()
