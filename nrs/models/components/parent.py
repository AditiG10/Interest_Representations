import torch
import torch.nn as nn
from typing import Tuple, Optional


class SingleTextParentRec(nn.Module):
    
    def __init__(self, 
        news_encoder: nn.Module, 
        user_encoder: nn.Module, 
        rec_model: nn.Module, 
        text_feature: str = 'title_emb'
    ):
        """
        Initializes the SingleTextParentRec model.

        Args:
            news_encoder (nn.Module): Encoder for news articles.
            user_encoder (nn.Module): Encoder for user features.
            rec_model (nn.Module): Recommendation model.
            text_feature (str, optional): Feature to use from the input. Defaults to 'title_emb'.
        """
        super(SingleTextParentRec, self).__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.rec_model = rec_model
        self.text_feature = text_feature

    def _forward(
        self, 
        history: Tuple[torch.Tensor], 
        candidates: Tuple[torch.Tensor],
        add_user_feats: Optional[Tuple[torch.Tensor]] = None,
        return_embeddings: bool = False
    ):
        """
        Forward pass for the SingleTextParentRec model.

        Args:
            history (Tuple[torch.Tensor]): User's history tensor.
            candidates (Tuple[torch.Tensor]): Candidate items tensor.
            add_user_feats (Optional[Tuple[torch.Tensor]], optional): Additional user features. Defaults to None.
            return_embeddings (bool, optional): Whether to return embeddings. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: Recommendation scores (and optionally user and candidate embeddings).
        """
        # TODO: make this more general
        h, hm = self.news_encoder(history)
        c, _ = self.news_encoder(candidates)
        u = self.user_encoder((h, hm), add_user_feats)
        r = self.rec_model(u, c)
        if return_embeddings:
            return r, u, c
        else:
            return r

    def forward(self, batch: dict, return_embeddings: bool = False):
        """
        Forward pass using batch data.

        Args:
            batch (Dict[str, Any]): Input batch containing user and candidate features.
            return_embeddings (bool, optional): Whether to return embeddings. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: Recommendation scores (and optionally user and candidate embeddings).
        """
        return self._forward(
            history=batch['user_features']['history'][self.text_feature],
            candidates=batch['candidate_features'][self.text_feature],
            add_user_feats=batch['user_features']['other'],
            return_embeddings=return_embeddings
        )
    

class MultiInterestParentRec(nn.Module):
    def __init__(self, 
                 news_encoder: nn.Module, 
                 user_encoder: nn.Module, 
                 rec_model: nn.Module, 
                 text_feature: str = 'title_emb'):
        """
        Initializes the MultiInterestParentRec model.

        Args:
            news_encoder (nn.Module): Encoder for news articles.
            user_encoder (nn.Module): Encoder for user features.
            rec_model (nn.Module): Recommendation model.
            text_feature (str, optional): Feature to use from the input. Defaults to 'title_emb'.
        """
        super(MultiInterestParentRec, self).__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.rec_model = rec_model
        self.text_feature = text_feature

    def _forward(self, 
                 history: Tuple[torch.Tensor], 
                 candidates: Tuple[torch.Tensor], 
                 add_user_feats: Optional[Tuple[torch.Tensor]] = None, 
                 return_embeddings: bool = False):
        """
        Forward pass for the MultiInterestParentRec model.

        Args:
            history (Tuple[torch.Tensor]): User's history tensor.
            candidates (Tuple[torch.Tensor]): Candidate items tensor.
            add_user_feats (Optional[Tuple[torch.Tensor]], optional): Additional user features. Defaults to None.
            return_embeddings (bool, optional): Whether to return embeddings. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: Maximum recommendation scores (and optionally user and candidate embeddings).
        """
        h, hm = self.news_encoder(history)
        c, _ = self.news_encoder(candidates)
        u = self.user_encoder((h, hm), add_user_feats)
        scores = self.rec_model(u, c)
        max_scores = torch.max(scores, dim=2, keepdim=True).values
        if return_embeddings:
            return max_scores, u, c
        else:
            return max_scores

    def forward(self, batch: dict, return_embeddings: bool = False):
        """
        Forward pass using batch data.

        Args:
            batch (Dict[str, Any]): Input batch containing user and candidate features.
            return_embeddings (bool, optional): Whether to return embeddings. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: Maximum recommendation scores (and optionally user and candidate embeddings).
        """
        return self._forward(
            history=batch['user_features']['history'][self.text_feature],
            candidates=batch['candidate_features'][self.text_feature],
            add_user_feats=batch['user_features'].get('other', None),
            return_embeddings=return_embeddings
        )


