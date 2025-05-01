from os import PathLike
import os
from os.path import join, exists
import torch
import torch.nn as nn
import random
from os import PathLike
import argparse
from nrs.models.components.user_encoding import UserEncoder
from nrs.models.components.multi_interest_user_encoder import MultiInterestUserEncoder


#ExampleforUserEncoderClass
# Generate dummy input data
batch_size = 10
seq_len = 20
emb_dim = 32
x = torch.randn(batch_size, seq_len, emb_dim)  # News embeddings
m = torch.ones(batch_size, seq_len, 1)  # Attention masks

# Create an instance of the UserEncoder
pooler = nn.MaxPool1d(kernel_size=3)
p_dropout = 0.1
emb_dim = 32
att = None 
head = True
activation = nn.ReLU()
bias = True

user_encoder = UserEncoder(pooler=pooler, p_dropout=p_dropout, emb_dim=emb_dim, att=att, head=head, activation=activation, bias=bias)

# Forward pass through the UserEncoder
user_representation = user_encoder(x)
print("Output shape:", user_representation.shape) # Output shape should be (batch_size, emb_dim)


'''

#ExampleForMultiInterestUserEncoderClass

# Define input tensors
batch_size = 2
seq_len = 5
emb_dim = 10
num_interests = 3

hist_features = torch.randn(batch_size, seq_len, emb_dim)
news_embeddings = torch.randn(batch_size, seq_len, emb_dim)
attention_masks = torch.randint(0, 2, (batch_size, seq_len))

# Instantiate the MultiInterestUserEncoder
pooler = nn.MaxPool1d(kernel_size=3)
p_dropout = 0.2
encoder = MultiInterestUserEncoder(pooler, p_dropout, num_interests, emb_dim)

# Forward pass
output = encoder(hist_features, news_embeddings, attention_masks)
print(output.shape)  # Output shape should be (batch_size, num_interests, emb_dim)
'''