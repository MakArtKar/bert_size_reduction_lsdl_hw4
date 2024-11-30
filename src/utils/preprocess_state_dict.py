from collections import OrderedDict
from typing import Dict

import torch.nn as nn


def preprocess_state_dict(state_dict: Dict[str, nn.Module], mapping: Dict[str, str]) -> Dict[str, nn.Module]:
    result = OrderedDict()
    for key, value in state_dict.items():
        for from_key, to_key in mapping.items():
            if key.startswith(from_key):
                key = to_key + key[len(from_key):]
        result[key] = value
    return result
