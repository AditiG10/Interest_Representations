from os import PathLike
import os
from os.path import join, exists
import torch
import torch.nn as nn
from nrs.models.components.parent import ParentRec
from nrs.models.components.news_encoding import TextEncoder, CategoryEncoder
from nrs.models.components.user_encoding import UserEncoder
from nrs.models.components.scoring import DotScoring


'''class DummyPooler(nn.Module):
    def __init__(self, emb_dim, activation, bias=True):
        super(DummyPooler, self).__init__()
        self.activation = activation
        self.bias = bias
        if emb_dim is not None:
            self.linear1 = nn.Linear(emb_dim, emb_dim, bias=bias)
            self.linear2 = nn.Linear(emb_dim, emb_dim, bias=bias)
            # Add a linear layer to resize input
            self.resize_linear = nn.Linear(emb_dim * 10, emb_dim)

    def forward(self, x, m):
        # Dummy pooling operation (just a placeholder)
        if hasattr(self, 'linear1'):
            x = self.activation(self.linear1(x))
            x = self.activation(self.linear2(x))

        # Resize the input tensor
        x = x.view(x.size(0), -1)  # Flatten along the last dimension
        x = self.resize_linear(x)

        return x'''


class DummyPooler(nn.Module):
    def __init__(self, emb_dim, activation, bias=True):
        super(DummyPooler, self).__init__()
        self.activation = activation
        self.bias = bias
        if emb_dim is not None:
            self.linear1 = nn.Linear(emb_dim, emb_dim, bias=bias)
            self.linear2 = nn.Linear(emb_dim, emb_dim, bias=bias)
            # Add a linear layer to resize input
            self.resize_linear = nn.Linear(emb_dim * 10, emb_dim)
            self.out_dim = emb_dim  # Add an out_dim attribute to store the output dimension

    def forward(self, x, m):
    # Dummy pooling operation (just a placeholder)
        if hasattr(self, 'linear1'):
            x = self.activation(self.linear1(x))
            x = self.activation(self.linear2(x))
        # Resize the input tensor
        x = x.view(x.size(0), x.size(1), -1)  # Flatten along the last dimension
        x = self.resize_linear(x)

        return x


# Instantiate the components
emb_dim = 128
activation = nn.ReLU()
pooler = DummyPooler(emb_dim=emb_dim, activation=activation)

# Instantiate the News Encoder
news_encoder = TextEncoder(pooler=pooler, p_dropout=0.1, out_features=128)


# Instantiate the components
emb_dim = 128
activation = nn.ReLU()
pooler = DummyPooler(emb_dim=emb_dim, activation=activation)
# Instantiate the News Encoder
# Instantiate the News Encoder
news_encoder = TextEncoder(pooler=DummyPooler(
    emb_dim=128, activation=nn.ReLU()), p_dropout=0.1, out_features=128)

# Instantiate the User Encoder
# Instantiate the User Encoder
user_encoder = UserEncoder(pooler=DummyPooler(emb_dim=128, activation=nn.ReLU()),
                           p_dropout=0.1, emb_dim=128, activation=nn.ReLU())


# Instantiate the Recommendation Model
rec_model = DotScoring()

# Instantiate the ParentRec Class
parent_rec = ParentRec(news_encoder=news_encoder,
                       user_encoder=user_encoder, rec_model=rec_model)


# Define batch size (b), sequence length (s), and embedding dimension (d)
b = 3
s = 10
d = 128

# Define the history length (nh) and number of candidates (nc)
nh = 3
nc = 5

# Construct the sample_batch dictionary
sample_batch = {
    'user_features': {
        'history': {
            'title_emb': (torch.randn((b, nh, s, d)), torch.ones((b, nh, s, 1)))
        },
        'other': torch.randn((b, 10))  # Assuming shape is (B, F) for other user features
    },
    'candidate_features': {
        'title_emb': (torch.randn((b, nc, s, d)), torch.ones((b, nc, s, 1)))
    }
}

# Pass the sample_batch to the ParentRec model
output = parent_rec(sample_batch, return_embeddings=False)
print(output.shape)  # Output shape of the recommendation scores