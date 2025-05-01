import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class AdditiveAttention(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(AdditiveAttention, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor = None, return_weights: bool = False):
        a = self.fc1(x)
        a = torch.tanh(a)
        a = self.fc2(a)
        a = torch.exp(a)
        if m is not None:
            a = a * m
        a = a / (torch.sum(a, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(a.transpose(-1, -2), x)
        if return_weights:
            return x, a
        else:
            return x


class MultiInterestUserEncoderSH(nn.Module):
    def __init__(self, 
                 p_dropout: float, 
                 num_interests: int, 
                 emb_dim: int, 
                 hidden_features: int, 
                 activation: nn.Module = nn.ReLU(), 
                 bias: bool = True):
        super(MultiInterestUserEncoderSH, self).__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.poolers = nn.ModuleList([
            AdditiveAttention(emb_dim, hidden_features) for _ in range(num_interests)
        ])
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim, bias=bias),
            activation,
            nn.Linear(emb_dim, emb_dim, bias=bias)
        )
        self.activation = activation

    def forward(self, inpt: Tuple[torch.Tensor, torch.Tensor], return_weights: bool = False):
        x, m = inpt
        x = self.dropout(x)
        x_new = []
        weights = []

        for i, pooler in enumerate(self.poolers):
            att_output, weight = pooler(x, m, return_weights=True)
            transformed_output = self.head(att_output)  # Use the single head here
            x_new.append(transformed_output)
            if return_weights:
                weights.append(weight)

        x_new = torch.cat(x_new, dim=1)  # Concatenate along the interest dimension
        if return_weights:
            weights = torch.cat(weights, dim=1)
            return x_new, weights
        return x_new


# Example usage
if __name__ == "__main__":
    # Assuming the dimensions for emb_dim, etc., are set appropriately.
    news_embeddings = torch.randn(1, 5, 10)  # (B:1, N:5, D:10)
    attention_masks = torch.ones(1, 5, 1)    # (B:1, N:5, 1)

    p_dropout = 0.1
    num_interests = 3
    emb_dim = 10
    hidden_features = 20
    activation = nn.ReLU()
    multi_interest_encoder = MultiInterestUserEncoderSH(p_dropout, num_interests, emb_dim, hidden_features, activation)

    # Forward pass
    pooled_representations, weights = multi_interest_encoder((news_embeddings, attention_masks), return_weights=True)

    # Calculate disentanglement loss
    #dis_loss_cosine = disentanglement_loss_cosine_similarity(pooled_representations)
   

    # Print shapes and results
    print("Initial News Embeddings Shape:", news_embeddings.shape)
    print("Initial Attention Masks Shape:", attention_masks.shape)
    print("Final Pooled Representations Shape:", pooled_representations.shape)
    
    #print("Disentanglement Loss:", dis_loss_cosine.item())
    #print()

    print("Pooled Representations:")
    print(pooled_representations)
    print("Weights:")
    print(weights)
