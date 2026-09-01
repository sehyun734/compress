import random

import torch
from datasets import load_dataset
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)


def load_llm(
    name: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype,
        device_map="cuda",
    ).eval()
    model.config.use_cache = False
    return model, tokenizer


def load_vlm(
    name: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, ProcessorMixin]:
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(
        name,
        dtype=dtype,
        device_map="cuda",
    ).eval()
    return model, processor


def load_calib(
    tokenizer: PreTrainedTokenizerBase,
    n_sample: int,
    seq_len: int,
    seed: int,
) -> list[Tensor]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tokenizer(
        "\n\n".join(ds["text"]),
        return_tensors="pt",
        verbose=False,  # seq_len으로 잘라 쓰므로 model_max_length 초과 경고 무의미
    ).input_ids
    gen = random.Random(seed)
    samples = []
    for _ in range(n_sample):
        i_start = gen.randint(0, ids.shape[1] - seq_len - 1)
        samples.append(ids[:, i_start : i_start + seq_len])
    return samples


def load_eval(
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
) -> list[Tensor]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer(
        "\n\n".join(ds["text"]),
        return_tensors="pt",
        verbose=False,  # seq_len으로 잘라 쓰므로 model_max_length 초과 경고 무의미
    ).input_ids
    n_sample = ids.shape[1] // seq_len
    return [ids[:, i * seq_len : (i + 1) * seq_len] for i in range(n_sample)]


def load_sft(
    tokenizer: PreTrainedTokenizerBase,
    n_sample: int,
    seq_len: int,
    seed: int,
) -> list[tuple[Tensor, Tensor]]:
    ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", streaming=True)
    ds = ds.filter(
        lambda row: row["domain"] == "math"
        and "</think>" in row["conversations"][1]["value"]
    )
    ds = ds.shuffle(seed=seed, buffer_size=n_sample)
    samples = []
    for row in ds.take(n_sample):
        prompt = row["conversations"][0]["value"]
        trace = row["conversations"][1]["value"]
        msgs = [{"role": "user", "content": prompt}]
        txt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        ids_prompt = tokenizer(
            txt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        ids_trace = tokenizer(
            trace.removeprefix("<think>") + tokenizer.eos_token,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        ids = torch.cat([ids_prompt, ids_trace], dim=-1)[:, :seq_len]
        labels = ids.clone()
        labels[:, : ids_prompt.shape[1]] = -100
        samples.append((ids, labels))
    return samples
