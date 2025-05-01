import torch.nn as nn

from ..components import SingleTextParentRec, TextEncoder, layers, UserEncoder, MultiInterestUserEncoder , MultiInterestParentRec
from ..components.layers import AdditiveAttention

class MeanRec(SingleTextParentRec):

    def __init__(self, cfg, rec_model: nn.Module):
        title_pooler = layers.MaskedMean()
        title_encoder = TextEncoder(
            att=None,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            in_features=cfg.d_backbone,
            out_features=cfg.title_emb_dim,
            bias=cfg.bias
        )
        hist_pooler = layers.MaskedMean()
        user_encoder = UserEncoder( 
            pooler=hist_pooler,
            att=None,
            head=False,
            p_dropout=cfg.p_dropout,
            emb_dim=cfg.title_emb_dim,
            bias=cfg.bias
        )
        super(MeanRec, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model
        )


class MeanRecMultiInterest(MultiInterestParentRec):
    def __init__(self, cfg, scoring_fn: nn.Module):
        # Title Encoder for news using MaskedMean for parameter-free pooling
        title_pooler = layers.MaskedMean()
        title_encoder = TextEncoder(
            att=None,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            in_features=cfg.d_backbone,
            out_features=cfg.title_emb_dim,
            bias=cfg.bias
        )

        # Multi-Interest User Encoder using AdditiveAttention for internal pooling
        user_encoder = MultiInterestUserEncoder(
            p_dropout=cfg.p_dropout,
            num_interests=cfg.num_interests,
            emb_dim=cfg.title_emb_dim,
            hidden_features=cfg.title_emb_dim,  # Hidden features should match the embedding dimension
            activation=nn.ReLU(),
            bias=cfg.bias
        )

        # Define the rec_model
        rec_model = scoring_fn

        super(MeanRecMultiInterest, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model,
            text_feature='title_emb'
        )

