from dataclasses import dataclass

from simple_parsing import parse

from shared.load import load_eval, load_llm, load_vlm
from shared.metrics import gqa, pope, ppl
from shared.utils import print_args


@dataclass
class Args:
    model: str = (
        "meta-llama/Llama-3.2-3B"
        # "DeepSeek-R1-Distill-Qwen-1.5B_reqat"
        # "llava-hf/llava-1.5-7b-hf"
    )
    seq_len: int = 2048
    vlm: bool = False
    limit: int = 32


def main() -> None:
    args = parse(Args)
    print_args(args)
    if args.vlm:
        model, _ = load_vlm(args.model)
        print(f"pope={pope(model, args.limit):.4f}")
        print(f"gqa={gqa(model, args.limit):.4f}")
    else:
        model, tokenizer = load_llm(args.model)
        print(f"ppl={ppl(model, load_eval(tokenizer, args.seq_len)):.4f}")


if __name__ == "__main__":
    main()
