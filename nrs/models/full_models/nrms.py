import torch
import torch.nn as nn
from typing import Tuple, Optional
from ..components import layers, news_encoding, parent
from ..components import user_encoding
from ..components.multi_interest_user_encoder import MultiInterestUserEncoder
from ...utils import collaps_mask
from ..components.parent import MultiInterestParentRec

class NRMS(parent.SingleTextParentRec):

    def __init__(self, cfg, rec_model: nn.Module):
        title_att = layers.MultiHeadAttention(
            n_heads=cfg.n_heads,
            d_model=cfg.d_backbone
        )
        title_pooler = layers.AdditiveAttention(
            in_features=cfg.d_backbone,
            hidden_features=256
        )
        title_encoder = news_encoding.TextEncoder(
            att=title_att,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            in_features=cfg.d_backbone,
            out_features=cfg.title_emb_dim
        )
        # ablation
        user_att = layers.MultiHeadAttention(
            n_heads=cfg.n_heads,
            d_model=cfg.title_emb_dim
        )
        user_pooler = layers.AdditiveAttention(
            in_features=cfg.title_emb_dim,
            hidden_features=256
        )
        user_encoder = user_encoding.UserEncoder(
            att=user_att,
            pooler=user_pooler,
            emb_dim=cfg.title_emb_dim,
            p_dropout=cfg.p_dropout,
            head=False
        )
        super(NRMS, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model
        )

class NRMS_LF(parent.SingleTextParentRec):
    '''NRSMS model with mean pooling user encoder ("late fusion")'''

    def __init__(self, cfg, rec_model: nn.Module):
        title_att = layers.MultiHeadAttention(
            n_heads=cfg.n_heads,
            d_model=cfg.d_backbone
        )
        title_pooler = layers.AdditiveAttention(
            in_features=cfg.d_backbone,
            hidden_features=256
        )
        title_encoder = news_encoding.TextEncoder(
            att=title_att,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            in_features=cfg.d_backbone,
            out_features=cfg.title_emb_dim
        )
        user_pooler = layers.MaskedMean()
        user_encoder = user_encoding.UserEncoder(
            att=None,
            pooler=user_pooler,
            emb_dim=cfg.title_emb_dim,
            p_dropout=cfg.p_dropout,
            head=False
        )
        super(NRMS_LF, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model
        )

#Multi-Interest Representation 
        
class MultiInterestNRMS(MultiInterestParentRec):
    def __init__(self, cfg, rec_model: nn.Module):
        # Title Encoder with Multi-Head Attention and Additive Attention
        title_att = layers.MultiHeadAttention(
            n_heads=cfg.n_heads,
            d_model=cfg.d_backbone
        )
        title_pooler = layers.AdditiveAttention(
            in_features=cfg.d_backbone,
            hidden_features=256
        )
        title_encoder = news_encoding.TextEncoder(
            att=title_att,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            in_features=cfg.d_backbone,
            out_features=cfg.title_emb_dim
        )

        # Multi-Interest User Encoder
        user_encoder = MultiInterestUserEncoder(
            p_dropout=cfg.p_dropout,
            num_interests=cfg.num_interests,
            emb_dim=cfg.title_emb_dim,
            hidden_features=256,
            activation=nn.ReLU(),
            bias=cfg.bias
        )

        # User Attention (Multi-Head Attention)
        user_attention = layers.MultiHeadAttention(
            n_heads=cfg.n_heads,
            d_model=cfg.title_emb_dim
        )

        super(MultiInterestNRMS, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model,
            text_feature='title_emb'
        )

        self.user_attention = user_attention

    def _forward(self, 
                 history: Tuple[torch.Tensor], 
                 candidates: Tuple[torch.Tensor], 
                 add_user_feats: Optional[Tuple[torch.Tensor]] = None, 
                 return_embeddings: bool = False):
        h, hm = self.news_encoder(history)
        c, _ = self.news_encoder(candidates)
        u = self.user_encoder((h, hm), add_user_feats)

        # Apply Multi-Head Attention to user interests
        u = self.user_attention(u, u, u)

        scores = self.rec_model(u, c)
        max_scores = torch.max(scores, dim=2, keepdim=True).values
        if return_embeddings:
            return max_scores, u, c
        else:
            return max_scores

