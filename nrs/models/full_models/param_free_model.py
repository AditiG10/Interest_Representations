import torch.nn as nn

from ..components import SingleTextParentRec, TextEncoder, layers, UserEncoder, MultiInterestUserEncoder , MultiInterestParentRec
from ..components.layers import AdditiveAttention

class ParamFreeRec(SingleTextParentRec):

    def __init__(self, cfg, rec_model: nn.Module):
        assert cfg.title_emb_dim == cfg.d_backbone
        title_pooler = layers.MaskedMean()
        title_encoder = TextEncoder(
            att=None,
            head=False,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            out_features=cfg.d_backbone
        )
        hist_pooler = layers.MaskedMean()
        user_encoder = UserEncoder(
            att=None,
            head=False,
            pooler=hist_pooler,
            p_dropout=cfg.p_dropout,
            emb_dim=cfg.title_emb_dim
        )
        super(ParamFreeRec, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model
        )



'''
Doubts: 
1. ParamFreeRecMultiInterest, MultiInterestUserEncoder already uses Additive attention for pooling but here in the user encoder above, MaskedMean is used. 
So what should be done? 

Notes:
Ensure that the title embedding dimension (cfg.title_emb_dim) matches the backbone dimension (cfg.d_backbone).
'''
class ParamFreeRecMultiInterest(MultiInterestParentRec):
    def __init__(self, cfg, scoring_fn: nn.Module):
        assert cfg.title_emb_dim == cfg.d_backbone
        
        # Title Encoder for news
        title_pooler = layers.MaskedMean()
        title_encoder = TextEncoder(
            att=None,
            head=False,
            pooler=title_pooler,
            p_dropout=cfg.p_dropout,
            out_features=cfg.d_backbone
        )

        # Multi-Interest User Encoder
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

        super(ParamFreeRecMultiInterest, self).__init__(
            news_encoder=title_encoder,
            user_encoder=user_encoder,
            rec_model=rec_model,
            text_feature='title_emb'
        )
