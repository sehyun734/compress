import math

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel


@torch.no_grad()
def ppl(model: PreTrainedModel, samples: list[Tensor]) -> float:
    device = next(model.parameters()).device
    nll = torch.zeros((), device=device)
    n_tok = 0
    for toks in samples:
        toks = toks.to(device)
        logits: Tensor = model(toks).logits[0, :-1].float()
        nll.add_(F.cross_entropy(logits, toks[0, 1:], reduction="sum"))
        n_tok += toks.shape[-1] - 1
    return math.exp(nll.item() / n_tok)
