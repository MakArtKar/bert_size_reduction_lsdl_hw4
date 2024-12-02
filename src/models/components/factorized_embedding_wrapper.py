import torch
from torch import nn
from transformers import AutoModel, AutoConfig


class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(FactorizedEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, embedding_size)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        x = self.projection(x)
        return x


def modify_bert_with_factorized_embedding(model, hidden_size):
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data
    U, S, Vh = torch.linalg.svd(embedding_matrix, full_matrices=False)
    U = U[:, :hidden_size]
    S = S[:hidden_size]
    Vh = Vh[:hidden_size, :]

    model.bert.embeddings.word_embeddings = FactorizedEmbedding(
        model.bert.embeddings.word_embeddings.num_embeddings,
        model.bert.embeddings.word_embeddings.embedding_dim,
        hidden_size,
    ).to(embedding_matrix.device)

    model.bert.embeddings.word_embeddings.token_embedding.weight.data = U
    model.bert.embeddings.word_embeddings.projection.weight.data = (torch.diag(S) @ Vh).T

    return model
