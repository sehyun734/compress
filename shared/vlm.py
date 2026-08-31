import torch.nn as nn
from transformers import PreTrainedModel


def get_proj(model: PreTrainedModel) -> nn.Module:
    match model.config.model_type:
        case "llava":
            return model.model.multi_modal_projector
        case _:
            raise NotImplementedError(model.config.model_type)


def get_img_id(model: PreTrainedModel) -> int:
    match model.config.model_type:
        case "llava":
            return model.config.image_token_id
        case _:
            raise NotImplementedError(model.config.model_type)


def get_lm(model: PreTrainedModel) -> nn.Module:
    match model.config.model_type:
        case "llava":
            return model.model.language_model
        case _:
            raise NotImplementedError(model.config.model_type)
