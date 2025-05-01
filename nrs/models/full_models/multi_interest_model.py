import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..components import layers
from ..components.layers import AdditiveAttention
from ..components.news_encoding import TextEncoder
from ..components.scoring import DotScoring

'''
class AdditiveAttention(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(AdditiveAttention, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor = None, return_weights: bool = False):
        # print(f"Input to AdditiveAttention x: {x.shape}")
        a = self.fc1(x)
        # print(f"After first linear layer (fc1) a: {a.shape}")
        a = torch.tanh(a)
        a = self.fc2(a)
        # print(f"After second linear layer (fc2) a: {a.shape}")
        a = torch.exp(a)
        if m is not None:
            a = a * m
            # print(f"After applying mask a: {a.shape}")
        a = a / (torch.sum(a, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(a.transpose(-1, -2), x)
        # print(f"Output of AdditiveAttention x: {x.shape}")
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
        x, m = inpt # x: (B, N, D), m: (B, N, 1)
        x = self.dropout(x)
        x_new = []
        weights = []

        for i, (pooler, head) in enumerate(zip(self.poolers, self.heads)):
            # print(f"Processing interest {i+1} with AdditiveAttention and corresponding head")
            att_output, weight = pooler(x, m, return_weights=True)
            # print(f"Output from AdditiveAttention {i+1}: {att_output.shape}")
            transformed_output = head(att_output)
            # print(f"Output from head {i+1}: {transformed_output.shape}")
            x_new.append(transformed_output)
            if return_weights:
                weights.append(weight)

        x_new = torch.cat(x_new, dim=1)
        if return_weights:
            weights = torch.cat(weights, dim=1)
            return x_new, weights
        return x_new

class MultiInterest(nn.Module):
    def __init__(self, cfg, scoring_fn):
        super(MultiInterest, self).__init__()
        self.news_encoder = TextEncoder(
            pooler=AdditiveAttention(cfg.d_backbone, cfg.title_emb_dim),
            p_dropout=cfg.p_dropout,
            out_features=cfg.title_emb_dim,
            in_features=cfg.d_backbone,
            head=True,
            activation=nn.ReLU(),
            bias=cfg.bias
        )
        
        self.user_encoder = MultiInterestUserEncoder(
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
        
        return max_scores,user_interests

    def forward(self, batch: dict):
        hist_title_features = batch['user_features']['history']['title_emb']
        cand_title_features = batch['candidate_features']['title_emb']
        return self._forward(hist_title_features, cand_title_features)
'''

class AdditiveAttention(nn.Module):
    def __init__(self, in_features, hidden_features, normalize_attention=True):
        super(AdditiveAttention, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 1)
        self.normalize_attention = normalize_attention

    def forward(self, x: torch.Tensor, m: torch.Tensor = None, return_weights: bool = False):
        a = self.fc1(x)
        a = torch.tanh(a)
        a = self.fc2(a)
        a = torch.exp(a)
        if m is not None:
            a = a * m
        if self.normalize_attention:
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
                 bias: bool = True,
                 normalize_attention: bool = True):
        super(MultiInterestUserEncoder, self).__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.poolers = nn.ModuleList([
            AdditiveAttention(emb_dim, hidden_features, normalize_attention) for _ in range(num_interests)
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
        x, m = inpt # x: (B, N, D), m: (B, N, 1)
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

class MultiInterest(nn.Module):
    def __init__(self, cfg, scoring_fn, normalize_attention=True):
        super(MultiInterest, self).__init__()
        self.news_encoder = TextEncoder(
            pooler=AdditiveAttention(cfg.d_backbone, cfg.title_emb_dim),  
            p_dropout=cfg.p_dropout,
            out_features=cfg.title_emb_dim,
            in_features=cfg.d_backbone,
            head=True,
            activation=nn.ReLU(),
            bias=cfg.bias
        )
        
        self.user_encoder = MultiInterestUserEncoder(
            p_dropout=cfg.p_dropout,
            num_interests=cfg.num_interests,
            emb_dim=cfg.user_emb_dim,
            hidden_features=256,
            activation=nn.ReLU(),
            bias=cfg.bias,
            normalize_attention=normalize_attention  # Pass the flag here
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
        
        return max_scores,user_interests,c_emb
    
    def forward(self, batch: dict):
        hist_title_features = batch['user_features']['history']['title_emb']
        cand_title_features = batch['candidate_features']['title_emb']
        return self._forward(hist_title_features, cand_title_features)
