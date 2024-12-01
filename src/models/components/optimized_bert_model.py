import torch
import torch.nn as nn


class SharedLayerBertEncoder(nn.Module):
    def __init__(self, encoder, group_length=1):
        super().__init__()
        layers = []
        for i in range(0, len(encoder.layer), group_length):
            layers.append(encoder.layer[i])
        self.layers = nn.ModuleList(layers)
        self.num_hidden_layers = len(encoder.layer)
        self.group_length = group_length

    def forward(self, hidden_states, attention_mask, **kwargs):
        """
        Forward pass through the shared encoder layers.
        Args:
            hidden_states (Tensor): Input embeddings of shape (batch_size, seq_len, hidden_size).
            attention_mask (Tensor): Attention mask of shape (batch_size, 1, 1, seq_len).
        Returns:
            Tensor: Output from the shared layers.
        """
        for i in range(self.num_hidden_layers):
            layer_outputs = self.layers[i // self.group_length](hidden_states, attention_mask, **kwargs)
            hidden_states = layer_outputs[0]  # Update hidden states

        return hidden_states


def modify_bert_with_weight_sharing(model: torch.nn.Module, group_length: int) -> torch.nn.Module:
    for i in range(len(model.bert.encoder.layer)):
        model.bert.encoder.layer[i] = model.bert.encoder.layer[i // group_length * group_length]
    return model
