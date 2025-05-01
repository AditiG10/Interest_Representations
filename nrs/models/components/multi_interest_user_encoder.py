import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

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

class MultiInterestUserEncoder(nn.Module):
    def __init__(self, 
                 p_dropout: float, 
                 num_interests: int, 
                 emb_dim: int, 
                 hidden_features: int, 
                 activation: nn.Module = nn.ReLU(), 
                 bias: bool = True):
        super(MultiInterestUserEncoder, self).__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.poolers = nn.ModuleList([
            AdditiveAttention(emb_dim, hidden_features) for _ in range(num_interests)
        ])
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim, bias=bias),
                activation,
                nn.Linear(emb_dim, emb_dim, bias=bias)
            ) for _ in range(num_interests)
        ])
        self.activation = activation

    def forward(self, inpt: Tuple[torch.Tensor, torch.Tensor], return_weights: bool = False):
        x, m = inpt
        x = self.dropout(x)
        x_new = []
        weights = []

        for i, (pooler, head) in enumerate(zip(self.poolers, self.heads)):
            att_output, weight = pooler(x, m, return_weights=True)
            transformed_output = head(att_output)
            x_new.append(transformed_output)
            if return_weights:
                weights.append(weight)

        x_new = torch.cat(x_new, dim=1)
        if return_weights:
            weights = torch.cat(weights, dim=1)
            return x_new, weights
        return x_new

#Cosine Similarity

def disentanglement_loss_cosine_similarity(user_embeddings):
    batch_size, num_interests, emb_dim = user_embeddings.size()

    # Normalize the embeddings
    user_embeddings = F.normalize(user_embeddings, p=2, dim=2)
    print("Normalized User Embeddings:", user_embeddings)

    # Calculate the dot product matrix (cosine similarity matrix)
    dot_product_matrix = torch.bmm(user_embeddings, user_embeddings.transpose(1, 2))
    print("Dot Product Matrix (Cosine Similarity):\n", dot_product_matrix)

    # Create an upper triangular mask excluding the diagonal
    mask = torch.triu(torch.ones_like(dot_product_matrix), diagonal=1)

    # Apply the mask to the dot product matrix
    dot_product_matrix = dot_product_matrix * mask

    # Calculate the disentanglement loss
    dis_loss = torch.sum(dot_product_matrix) / (batch_size * (num_interests * (num_interests - 1)) / 2)

    return dis_loss


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
    multi_interest_encoder = MultiInterestUserEncoder(p_dropout, num_interests, emb_dim, hidden_features, activation)

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
