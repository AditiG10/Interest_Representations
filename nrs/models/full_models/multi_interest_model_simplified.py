import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..components import layers
from ..components.layers import AdditiveAttention
from ..components.news_encoding import TextEncoder
from ..components.scoring import DotScoring

class MultiInterestUserEncoder_Simplified(nn.Module):
    def __init__(self, 
                 p_dropout: float, 
                 num_interests: int, 
                 emb_dim: int, 
                 hidden_features: int, 
                 activation: nn.Module = nn.ReLU(), 
                 bias: bool = True):
        super(MultiInterestUserEncoder_Simplified, self).__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.poolers = nn.ModuleList([
            AdditiveAttention(emb_dim, hidden_features) for _ in range(num_interests)
        ])
        self.head = nn.Sequential(
            nn.Linear(emb_dim * num_interests, hidden_features, bias=bias),
            activation,
            nn.Linear(hidden_features, emb_dim * num_interests, bias=bias)
        )
        self.activation = activation  

    def forward(self, inpt: Tuple[torch.Tensor, torch.Tensor], return_weights: bool = False):
        x, m = inpt  # x: (B, N, D), m: (B, N, 1)
        x = self.dropout(x)
        pooled_outputs = []
        weights = []

        for i, pooler in enumerate(self.poolers):
            att_output, weight = pooler(x, m, return_weights=True)
            pooled_outputs.append(att_output)
            if return_weights:
                weights.append(weight)

        # Concatenate pooled outputs along the interest dimension
        x_new = torch.cat(pooled_outputs, dim=1)  # (B, num_interests * D)
        
        # Flatten the concatenated output
        x_new = x_new.view(x_new.size(0), -1)  # (B, num_interests * D)
        
        # Pass through the single head
        transformed_output = self.head(x_new)
        
        # Reshape to (B, num_interests, D)
        transformed_output = transformed_output.view(-1, len(self.poolers), x.size(-1))
        
        if return_weights:
            weights = torch.cat(weights, dim=1)
            return transformed_output, weights
        return transformed_output


class MultiInterest_Simplified(nn.Module):
    def __init__(self, cfg, scoring_fn):
        super(MultiInterest_Simplified, self).__init__()
        self.news_encoder = TextEncoder(
            pooler=AdditiveAttention(cfg.d_backbone, cfg.title_emb_dim),
            p_dropout=cfg.p_dropout,
            out_features=cfg.title_emb_dim,
            in_features=cfg.d_backbone,
            head=True,
            activation=nn.ReLU(),
            bias=cfg.bias
        )
        
        self.user_encoder = MultiInterestUserEncoder_Simplified(
            p_dropout=cfg.p_dropout,
            num_interests=cfg.num_interests,
            emb_dim=cfg.user_emb_dim,
            hidden_features=256,  
            activation=nn.ReLU(),
            bias=cfg.bias
        )
        self.rec_model = scoring_fn

    def _forward(self, hist_news_features, cand_news_features):

        hist_emb, hist_mask = hist_news_features
        cand_emb, cand_mask = cand_news_features

        h_emb, h_masks = self.news_encoder((hist_emb, hist_mask))
        c_emb, c_masks = self.news_encoder((cand_emb, cand_mask))

        user_interests = self.user_encoder((h_emb, h_masks))

        scores = self.rec_model(user_interests, c_emb)
        max_scores = torch.max(scores, dim=2, keepdim=True).values
        
        return max_scores, user_interests

    def forward(self, batch: dict):
        hist_title_features = batch['user_features']['history']['title_emb']
        cand_title_features = batch['candidate_features']['title_emb']
        return self._forward(hist_title_features, cand_title_features)
