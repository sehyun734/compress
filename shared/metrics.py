import math

import torch
import torch.nn.functional as F
from lmms_eval.evaluator import simple_evaluate
from lmms_eval.models.simple.llava_hf import VICUNA_CHAT_TEMPLATE, LlavaHf, model_map
from lmms_eval.tasks import TaskManager
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


@torch.no_grad()
def pope(model: PreTrainedModel, n_eval: int) -> float:
    cls = model_map[model.config.model_type]
    orig = cls.from_pretrained
    cls.from_pretrained = lambda *a, **kw: model
    try:
        lm = LlavaHf(
            pretrained=model.name_or_path,
            batch_size=1,
            chat_template=VICUNA_CHAT_TEMPLATE,
        )
    finally:
        cls.from_pretrained = orig
    res = simple_evaluate(
        model=lm,
        tasks=["pope"],
        batch_size=1,
        limit=n_eval or None,
        task_manager=TaskManager(model_name="llava_hf"),
        log_samples=False,
    )
    return res["results"]["pope"]["pope_f1_score,none"]


@torch.no_grad()
def gqa(model: PreTrainedModel, n_eval: int = 32) -> float:
    cls = model_map[model.config.model_type]
    orig = cls.from_pretrained
    cls.from_pretrained = lambda *a, **kw: model
    try:
        lm = LlavaHf(
            pretrained=model.name_or_path,
            batch_size=1,
            chat_template=VICUNA_CHAT_TEMPLATE,
        )
    finally:
        cls.from_pretrained = orig
    res = simple_evaluate(
        model=lm,
        tasks=["gqa"],
        batch_size=1,
        limit=n_eval or None,
        task_manager=TaskManager(model_name="llava_hf"),
        log_samples=False,
    )
    return res["results"]["gqa"]["exact_match,none"]
