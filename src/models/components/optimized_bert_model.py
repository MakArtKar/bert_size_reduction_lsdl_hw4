import torch
import torch.nn as nn


def modify_bert_with_weight_sharing(model: torch.nn.Module, num_layers: int, in_a_row=True) -> torch.nn.Module:
    for i in range(len(model.bert.encoder.layer)):
        if in_a_row:
            model.bert.encoder.layer[i] = model.bert.encoder.layer[i // num_layers * num_layers]
        else:
            model.bert.encoder.layer[i] = model.bert.encoder.layer[i % num_layers]
    return model
