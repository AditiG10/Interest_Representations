from os import PathLike
import os
from os.path import join, exists
import torch
import torch.nn as nn
import random
from os import PathLike
import argparse
from nrs.models.components.news_encoding import TextEncoder, CategoryEncoder


# Define a dummy pooler function
class DummyPooler(nn.Module):
    def forward(self, x, m):
        return torch.mean(x, dim=1)

# Instantiate the TextEncoder
pooler = DummyPooler()
text_encoder = TextEncoder(pooler, p_dropout=0.1, out_features=128)

# Define input tensors
x = torch.randn(2, 3, 4, 768)  # Batch size 2, 3 news items, each with 4 sentences, each sentence with 768 features
m = torch.ones(2, 3, 4, 1)     # Mask for attention

# Forward pass
output = text_encoder((x, m))
print(output[0].shape)  # Output shape of the news embeddings
print(output[1].shape)  # Output shape of the attention masks


# Instantiate the CategoryEncoder
category_encoder = CategoryEncoder(n_categories=10, embedding_dim=32)

# Define input tensor
x = torch.LongTensor([1, 5, 7, 3, 0])  # Sample category indices

# Forward pass
output = category_encoder(x)
print(output.shape)  # Output shape of the category embeddings

