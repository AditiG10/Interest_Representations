import torch.nn as nn
import torch
from ..components import SingleTextParentRec, TextEncoder, layers, UserEncoder, MultiInterestUserEncoder , MultiInterestParentRec
from ..components.layers import AdditiveAttention


class BaseRec(SingleTextParentRec):

    def __init__(self, cfg, rec_model: nn.Module):
        title_pooler = layers.AdditiveAttention(
            in_features=cfg.d_backbone,
            hidden_features=256
        )
        title_encoder = TextEncoder(
            att=None,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            in_features=cfg.d_backbone,
            out_features=cfg.title_emb_dim,
            bias=cfg.bias
        )
        hist_pooler = layers.AdditiveAttention(
            in_features=cfg.title_emb_dim,
            hidden_features=256
        )
        user_encoder = UserEncoder(
            pooler=hist_pooler,
            att=None,
            head=False,
            p_dropout=cfg.p_dropout,
            emb_dim=cfg.title_emb_dim
        )
        super(BaseRec, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model
        )


class BaseRecMultiInterest(MultiInterestParentRec):
    def __init__(self, cfg, scoring_fn: nn.Module):
        # Title Encoder for news
        title_pooler = AdditiveAttention(
            in_features=cfg.d_backbone,
            hidden_features=256
        )
        title_encoder = TextEncoder(
            att=None,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            in_features=cfg.d_backbone,
            out_features=cfg.title_emb_dim,
            bias=cfg.bias
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

        # Define the rec_model
        rec_model = scoring_fn

        super(BaseRecMultiInterest, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model,
            text_feature='title_emb'
        )
