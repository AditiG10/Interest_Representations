import torch
import torch.nn as nn
from torch.nn import utils
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
import wandb
from abc import ABC, abstractmethod
import os
from os.path import join, exists
from pandas import DataFrame
import datetime
from typing import Optional
import pickle

import torch
import wandb 
import yaml
import os
import pickle
from os.path import join, exists
import random
from os import PathLike
import argparse
from dotmap import DotMap
from nrs.models import make_model
from nrs.data.mind import make_mind_data
from nrs import training
import psutil
from .evaluation import metrics as eval
from . import utils
from .utils import batch_to_device  
#from .evaluation import metrics_pytorch as eval

class BaseTrainer(ABC):

    def __init__(self, 
            cfg: DictConfig, 
            model: nn.Module, 
            trainset: Dataset, 
            testset: Dataset
        ):
        self.cfg = cfg
        self.model = model
        self._init_dataloaders(trainset, testset)

        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self._init_loss()

        self.current_epoch = 0
        self.current_train_step = 0
        self.current_test_step = 0

    @abstractmethod
    def _init_loss(self):
        raise NotImplementedError(
            'BaseTrainer needs to be subclassed with a loss implementation'
        )

    def _init_dataloaders(self, trainset: Dataset, testset: Dataset):
        self.trainloader = DataLoader(
            dataset=trainset,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle_data,
            drop_last=True,
            num_workers=self.cfg.num_workers
        )
        self.testloader = DataLoader(
            dataset=testset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=0  # self.cfg.num_workers
        )

    def forward(self, batch) -> torch.Tensor:
        '''implement output activation here if needed'''
        pass
    
    def _save_checkpoint(self):
        print('saving checkpoint')
        ckpt_path = join(self.cfg.dir, self.cfg.name, 'checkpoints')
        if not exists(ckpt_path):
            os.makedirs(ckpt_path)
        ckpt_dict = {
            'config': dict(self.cfg),
            'model_name': self.cfg.name,
            'state_dict': self.model.state_dict()
        }
        torch.save(ckpt_dict, join(ckpt_path, f'ckpt_{self.current_epoch}'))

    def _save_scores(self, targets: np.array, scores: np.array, stats: Optional[dict] = None):
        print('saving scores')
        scores_path = join(self.cfg.dir, self.cfg.name, 'predictions')
        if not exists(scores_path):
            os.makedirs(scores_path)
        scores_dict = {
            'targets': targets,
            'scores': scores,
            'stats': stats
        }
        torch.save(scores_dict, join(scores_path, f'predictions_{self.current_epoch}'))

    def _train_step(self, batch: dict) -> dict:
        self.optimizer.zero_grad()
        t = batch['targets'].to(self.device)
        s = self.forward(batch)
        if 'weights' in batch.keys():
            w = batch['weights'].to(self.device)
            loss = self.L(s, t, w)
        else:
            loss = self.L(s, t)
        loss.backward()
        self.optimizer.step()
        results = {
            'loss': loss.detach().cpu().numpy(),
            'logits': s
        }
        return results

    @abstractmethod
    def _test_step(self, batch: dict) -> dict:
        raise NotImplementedError('BaseTrainer needs to be subclassed with a _test_step implementation')

    def _train_iteration(self) -> dict:
        self.model.train()
        iteration_results = []
        for batch in tqdm(self.trainloader):
            results = self._train_step(batch)
            iteration_results.append(results)
            self._after_train_step(results)
        self._after_train_iteration(iteration_results)
        return iteration_results

    def _test_iteration(self) -> list:
        self.model.eval()
        iteration_results = []
        for batch in tqdm(self.testloader):
            results = self._test_step(batch)
            self._after_test_step(results)
            iteration_results.append(results)
        self._after_test_iteration(iteration_results)
        return iteration_results

    def train(self):
        # TODO: evetually switch to logging and testing in step-wise periods?
        for e in range(self.cfg.n_epochs):
            self.current_epoch = e
            print(f'\n Epoch {e}:')
            print('training:')
            train_results = self._train_iteration()
            if (e + 1) % self.cfg.test_freq == 0\
                or e == self.cfg.n_epochs - 1:
                print('testing:')
                test_results = self._test_iteration()
        if self.cfg.n_epochs == 0:
            test_results = self._test_iteration()
        # self._after_training(
        #     last_train_results=train_results,
        #     last_test_results=test_results
        # )

    def _after_train_step(self, results: dict):
        pass

    def _after_test_step(self, results: dict):
        pass

    def _after_train_iteration(self, results: dict):
        if self.cfg.ckpt_freq is not None:
            if self.current_epoch % self.cfg.ckpt_freq == 0\
                or self.current_epoch == self.cfg.n_epochs - 1:
                self._save_checkpoint()
        epoch_loss = np.mean([d['loss'] for d in results])
        if self.cfg.wandb:
            wandb.log({
                'train_loss': epoch_loss, 
                'epoch': self.current_epoch
                })
        print('train loss: ', epoch_loss)

    def _after_test_iteration(self, results: list):
        pass

    def _after_training(self, last_train_results: list, last_test_results: list):
        pass


class RankingTrainer(BaseTrainer):

    @torch.no_grad()
    def _test_step(self, batch: dict) -> dict:
        # TODO combine this with train step in forward
        t = batch['targets'].to(self.device)
        s = self.forward(batch)
        if 'weights' in batch.keys():
            w = batch['weights'].to(self.device)
            loss = self.L(s, t, w)
        else:
            loss = self.L(s, t)
        loss = loss.cpu().numpy()
        t = t.cpu().squeeze().numpy()
        s = s.cpu().squeeze().numpy()
        return_dict = {
            'ndcg@5': eval.ndcg_score(t, s, k=5),
            'ndcg@10': eval.ndcg_score(t, s, k=10),
            'rr': eval.rr_score(t, s),
            'ctr@1': eval.ctr_score(t, s, k=1),
            'auc': eval.auc_score(t, s),
            'acc': eval.acc_score(t, s),
            'rec': eval.recall_score(t, s), 
            'prec': eval.precision_score(t, s),
            'conf': eval.confusion_matrix(t, s),
            'scores': s,
            'targets': t,
            'loss': loss
        }
        self._after_test_step(return_dict)
        return return_dict

    def _after_test_iteration(self, results: list):
        loss = np.array([d['loss'] for d in results])
        ndcg5 = np.array([d['ndcg@5'] for d in results])
        ndcg10 = np.array([d['ndcg@10'] for d in results])
        rr = np.array([d['rr'] for d in results])
        ctr = np.array([d['ctr@1'] for d in results])
        acc = np.array([d['acc'] for d in results])
        auc = np.array([d['auc'] for d in results])
        rec = np.array([d['rec'] for d in results])
        prec = np.array([d['prec'] for d in results])
        epoch_loss = np.mean(loss)
        epoch_ndcg5 = np.mean(ndcg5)
        epoch_ndcg10 = np.mean(ndcg10)
        epoch_mrr = np.mean(rr)
        epoch_acc = np.mean(acc)
        epoch_auc = np.mean(auc)
        epoch_rec = np.mean(rec)
        epoch_prec = np.mean(prec)
        epoch_ctr = np.mean(ctr)
        epoch_conf = np.sum([d['conf'] for d in results], axis=0)
        scores = np.concatenate([d['scores'] for d in results])
        targets = np.concatenate([d['targets'] for d in results])
        score_hist = wandb.Histogram(scores, num_bins=50)
        conf_table = wandb.Table(
            columns=['0', '1'],
            data=epoch_conf.tolist()
            )
        self._save_scores(
            targets=targets, 
            scores=scores,
            stats={
                'auc': auc,
                'mrr': rr,
                'ndcg@5': ndcg5,
                'ndcg@10': ndcg10
            })
        if self.cfg.wandb:
            wandb.log({
                'test_loss': epoch_loss,
                'ndcg@5': epoch_ndcg5,
                'ndcg@10': epoch_ndcg10,
                'mrr': epoch_mrr,
                'ctr@1': epoch_ctr,
                'auc': epoch_auc,
                'acc': epoch_acc,
                'rec': epoch_rec,
                'prec': epoch_prec,
                'conf': conf_table,
                'scores': score_hist,
                'epoch': self.current_epoch
                })
        print(f'test loss: {epoch_loss:.4f}, ndcg@5: {epoch_ndcg5:.4f}, '\
                + f'mrr: {epoch_mrr:.4f}, ctr@1: {epoch_ctr:.4f}, auc: {epoch_auc:.4f}, '\
                + f'acc: {epoch_acc:.4f}, rec: {epoch_rec:.4f}, prec: {epoch_prec:.4f}'
            )

class BCERankingTrainer(RankingTrainer):

    def _init_loss(self):
        self.L = nn.BCELoss()

    def forward(self, batch) -> torch.Tensor:
        '''implement output activation here if needed'''
        return torch.sigmoid(self.model.forward(batch))


class BCELogitsRankingTrainer(RankingTrainer):

    def _init_loss(self):
        self.L = nn.functional.binary_cross_entropy_with_logits

    def forward(self, batch) -> torch.Tensor:
        '''no output activation applied here'''
        batch_to_device(batch, self.device)
        return self.model.forward(batch)

    @torch.no_grad()
    def _test_step(self, batch: dict) -> dict:
        '''need to add sigmoid after loss computation here'''
        t = batch['targets'].to(self.device)
        s = self.forward(batch)
        if 'weights' in batch.keys():
            w = batch['weights'].to(self.device)
            loss = self.L(s, t, w)
        else:
            loss = self.L(s, t)
        loss = loss.cpu().numpy()
        t = t.squeeze().cpu().numpy()
        # adding sigmoid to compute test metrics
        s = torch.sigmoid(s).squeeze().cpu().numpy()
        return_dict = {
            'ndcg@5': eval.ndcg_score(t, s, k=5),
            'ndcg@10': eval.ndcg_score(t, s, k=10),
            'rr': eval.rr_score(t, s),
            'ctr@1': eval.ctr_score(t, s, k=1),
            'auc': eval.auc_score(t, s),
            'acc': eval.acc_score(t, s),
            'rec': eval.recall_score(t, s), 
            'prec': eval.precision_score(t, s),
            'conf': eval.confusion_matrix(t, s),
            'scores': s,
            'targets': t,
            'loss': loss
        }
        self._after_test_step(return_dict)
        return return_dict 

class MSERankingTrainer(RankingTrainer):
    
    def _init_loss(self):
        def loss(prediction, target, weight: Optional[torch.tensor] = None):
            if weight is not None:
                l = nn.functional.mse_loss(prediction, target, reduction='none')
                l = l * weight
                return torch.mean(l)
            else:
                return nn.functional.mse_loss(prediction, target)
        self.L = loss

    def forward(self, batch) -> torch.Tensor:
        '''implement output activation here if needed'''
        batch_to_device(batch, self.device)
        oupt = self.model.forward(batch)
        return torch.relu(oupt)
        # return self.model.forward(batch)


# Disentanglement loss using cosine similarity
def disentanglement_loss_cosine(user_embeddings):
    batch_size, num_interests, emb_dim = user_embeddings.size()
    user_embeddings = nn.functional.normalize(user_embeddings, p=2, dim=2)  # Normalize embeddings
    dot_product_matrix = torch.bmm(user_embeddings, user_embeddings.transpose(1, 2))  # Cosine similarity matrix
    mask = torch.triu(torch.ones_like(dot_product_matrix), diagonal=1)  # Upper triangular mask
    dot_product_matrix = dot_product_matrix * mask  # Apply the mask
    dot_product_matrix = torch.abs(dot_product_matrix)  # Absolute values
    dis_loss = torch.sum(dot_product_matrix) / (batch_size * (num_interests * (num_interests - 1)) / 2)  # Disentanglement loss calculation
    return dis_loss


# Disentanglement loss using dot product
def disentanglement_loss_dot(user_embeddings):
    batch_size, num_interests, emb_dim = user_embeddings.size()
    dot_product_matrix = torch.bmm(user_embeddings, user_embeddings.transpose(1, 2))  # Dot product matrix
    mask = torch.triu(torch.ones_like(dot_product_matrix), diagonal=1)  # Upper triangular mask to ignore diagonal and lower triangle
    dot_product_matrix = dot_product_matrix * mask  # Apply mask
    dis_loss = torch.sum(dot_product_matrix) / (batch_size * (num_interests * (num_interests - 1)) / 2)  # Average the upper triangle values
    dis_loss = torch.abs(dis_loss)  # Take the absolute value of the loss
    return dis_loss



def pad_tensor(tensor, target_size):
    padding = target_size - tensor.size(1)
    if padding > 0:
        return torch.nn.functional.pad(tensor, (0, 0, 0, padding), mode='constant', value=0)
    return tensor

'''
class RankingTrainerSimplified(RankingTrainer):
    def __init__(self, cfg, model, trainset, testset, disentanglement_loss_fn=None):
        super().__init__(cfg, model, trainset, testset)
        weight_decay_value = float(getattr(self.cfg, 'weight_decay', 1e-4))
        self.weight_decay = torch.tensor(weight_decay_value).float().to(self.device)
        self.lambda_s = torch.tensor(getattr(self.cfg, 'lambda_s', 0.5)).float().to(self.device)
        self.lambda_u = 1.0 - self.lambda_s

        self.user_representations = []
        self.candidate_representations = []

        if disentanglement_loss_fn == 'cosine':
            self.disentanglement_loss_fn = disentanglement_loss_cosine
        elif disentanglement_loss_fn == 'dot':
            self.disentanglement_loss_fn = disentanglement_loss_dot
        elif disentanglement_loss_fn is not None:
            raise ValueError(f"Unknown disentanglement loss function: {disentanglement_loss_fn}")
        else:
            self.disentanglement_loss_fn = None

    def _train_step(self, batch: dict) -> dict:
        self.optimizer.zero_grad()
        t = batch['targets'].to(self.device)

        # Expect model to return (scores, user_interests, candidate_embeddings)
        s, user_interests, candidate_embeddings = self.model.forward(batch)

        if torch.isnan(s).any() or (user_interests is not None and torch.isnan(user_interests).any()):
            raise ValueError("NaNs detected in the model's output")

        score_loss = self.L(s, t) if 'weights' not in batch else self.L(s, t, batch['weights'].to(self.device))
        dis_loss = self.disentanglement_loss_fn(user_interests) if self.disentanglement_loss_fn is not None else torch.tensor(0.0, device=self.device)
        if torch.isnan(score_loss).any() or torch.isnan(dis_loss).any():
            raise ValueError("NaNs detected in the loss calculation")

        l2_reg = sum(torch.norm(param, 2) for param in self.model.parameters()).sum()
        total_loss = self.lambda_s * score_loss + self.lambda_u * dis_loss + self.weight_decay * l2_reg
        if torch.isnan(total_loss).any():
            raise ValueError("NaNs detected in the total loss")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if user_interests is not None:
            self.user_representations.append(user_interests.cpu())
        if candidate_embeddings is not None and len(self.candidate_representations) == 0:
            self.candidate_representations.append(candidate_embeddings.cpu())

        return {
            'loss': total_loss.detach().cpu().numpy(),
            'score_loss': score_loss.detach().cpu().numpy(),
            'dis_loss': dis_loss.detach().cpu().numpy() if self.disentanglement_loss_fn is not None else None,
            'logits': s
        }

    @torch.no_grad()
    def _test_step(self, batch: dict) -> dict:
        t = batch['targets'].to(self.device)

        # Expect model to return (scores, user_interests, candidate_embeddings)
        s, user_interests, candidate_embeddings = self.model.forward(batch)

        if torch.isnan(s).any() or (user_interests is not None and torch.isnan(user_interests).any()):
            raise ValueError("NaNs detected in the model's output during testing")

        loss = self.L(s, t) if 'weights' not in batch else self.L(s, t, batch['weights'].to(self.device))

        if user_interests is not None:
            self.user_representations.append(user_interests.cpu())
        if candidate_embeddings is not None and len(self.candidate_representations) == 0:
            self.candidate_representations.append(candidate_embeddings.cpu())

        return_dict = {
            'ndcg@5': eval.ndcg_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy(), k=5),
            'ndcg@10': eval.ndcg_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy(), k=10),
            'rr': eval.rr_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy()),
            'ctr@1': eval.ctr_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy(), k=1),
            'auc': eval.auc_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy()),
            'acc': eval.acc_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy()),
            'rec': eval.recall_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy()),
            'prec': eval.precision_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy()),
            'conf': eval.confusion_matrix(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy()),
            'scores': s.cpu().squeeze().numpy(),
            'targets': t.cpu().squeeze().numpy(),
            'loss': loss.cpu().numpy()
        }

        self._after_test_step(return_dict)
        return return_dict

    def get_user_representations(self):
        return torch.cat(self.user_representations, dim=0) if self.user_representations else None

    def get_candidate_representations(self):
        return torch.cat(self.candidate_representations, dim=0) if self.candidate_representations else None

    def save_user_representations(self, path):
        if self.user_representations:
            all_users = torch.cat(self.user_representations, dim=0)
            with open(path, 'wb') as f:
                pickle.dump(all_users.detach().cpu().numpy(), f)
            print(f"User representations saved to {path}")
        else:
            print("No user representations to save.")

    def save_candidate_representations(self, path):
        if self.candidate_representations:
            all_candidates = torch.cat(self.candidate_representations, dim=0)
            with open(path, 'wb') as f:
                pickle.dump(all_candidates.detach().cpu().numpy(), f)
            print(f"Candidate representations saved to {path}")
        else:
            print("No candidate representations to save.")

class BCELogitsRankingTrainerSimplified(RankingTrainerSimplified):
    def _init_loss(self):
        self.L = nn.functional.binary_cross_entropy_with_logits

    def forward(self, batch) -> torch.Tensor:
        batch_to_device(batch, self.device)
        return self.model.forward(batch)

    def _train_step(self, batch: dict) -> dict:
        results = super()._train_step(batch)
        results['logits'] = torch.sigmoid(results['logits'].clone().detach())
        return results

    @torch.no_grad()
    def _test_step(self, batch: dict) -> dict:
        results = super()._test_step(batch)
        results['scores'] = torch.sigmoid(torch.tensor(results['scores']))
        return results
'''

class RankingTrainerSimplified(RankingTrainer):
    def __init__(self, cfg, model, trainset, testset, disentanglement_loss_fn=None, save_representations=True):
        super().__init__(cfg, model, trainset, testset)
        self.save_representations = save_representations
        weight_decay_value = float(getattr(self.cfg, 'weight_decay', 1e-4))
        self.weight_decay = torch.tensor(weight_decay_value).float().to(self.device)
        self.lambda_s = torch.tensor(getattr(self.cfg, 'lambda_s', 0.5)).float().to(self.device)
        self.lambda_u = 1.0 - self.lambda_s

        self.user_representations = []
        self.candidate_representations = []

        if disentanglement_loss_fn == 'cosine':
            self.disentanglement_loss_fn = disentanglement_loss_cosine
        elif disentanglement_loss_fn == 'dot':
            self.disentanglement_loss_fn = disentanglement_loss_dot
        elif disentanglement_loss_fn is not None:
            raise ValueError(f"Unknown disentanglement loss function: {disentanglement_loss_fn}")
        else:
            self.disentanglement_loss_fn = None

    def _train_step(self, batch: dict) -> dict:
        self.optimizer.zero_grad()
        t = batch['targets'].to(self.device)

        # Expect model to return (scores, user_interests, candidate_embeddings)
        s, user_interests, candidate_embeddings = self.model.forward(batch)

        if torch.isnan(s).any() or (user_interests is not None and torch.isnan(user_interests).any()):
            raise ValueError("NaNs detected in the model's output")

        score_loss = self.L(s, t) if 'weights' not in batch else self.L(s, t, batch['weights'].to(self.device))
        dis_loss = self.disentanglement_loss_fn(user_interests) if self.disentanglement_loss_fn is not None else torch.tensor(0.0, device=self.device)
        if torch.isnan(score_loss).any() or torch.isnan(dis_loss).any():
            raise ValueError("NaNs detected in the loss calculation")

        l2_reg = sum(torch.norm(param, 2) for param in self.model.parameters()).sum()
        total_loss = self.lambda_s * score_loss + self.lambda_u * dis_loss + self.weight_decay * l2_reg
        if torch.isnan(total_loss).any():
            raise ValueError("NaNs detected in the total loss")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if user_interests is not None:
            self.user_representations.append(user_interests.cpu())
        if candidate_embeddings is not None and len(self.candidate_representations) == 0:
            self.candidate_representations.append(candidate_embeddings.cpu())

        return {
            'loss': total_loss.detach().cpu().numpy(),
            'score_loss': score_loss.detach().cpu().numpy(),
            'dis_loss': dis_loss.detach().cpu().numpy() if self.disentanglement_loss_fn is not None else None,
            'logits': s
        }

    @torch.no_grad()
    def _test_step(self, batch: dict) -> dict:
        t = batch['targets'].to(self.device)

        # Expect model to return (scores, user_interests, candidate_embeddings)
        s, user_interests, candidate_embeddings = self.model.forward(batch)

        if torch.isnan(s).any() or (user_interests is not None and torch.isnan(user_interests).any()):
            raise ValueError("NaNs detected in the model's output during testing")

        loss = self.L(s, t) if 'weights' not in batch else self.L(s, t, batch['weights'].to(self.device))

        if user_interests is not None:
            self.user_representations.append(user_interests.cpu())
        if candidate_embeddings is not None and len(self.candidate_representations) == 0:
            self.candidate_representations.append(candidate_embeddings.cpu())

        return_dict = {
            'ndcg@5': eval.ndcg_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy(), k=5),
            'ndcg@10': eval.ndcg_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy(), k=10),
            'rr': eval.rr_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy()),
            'ctr@1': eval.ctr_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy(), k=1),
            'auc': eval.auc_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy()),
            'acc': eval.acc_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy()),
            'rec': eval.recall_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy()),
            'prec': eval.precision_score(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy()),
            'conf': eval.confusion_matrix(t.cpu().squeeze().numpy(), s.cpu().squeeze().numpy()),
            'scores': s.cpu().squeeze().numpy(),
            'targets': t.cpu().squeeze().numpy(),
            'loss': loss.cpu().numpy()
        }

        self._after_test_step(return_dict)
        return return_dict

    def get_user_representations(self):
        return torch.cat(self.user_representations, dim=0) if self.user_representations else None

    def get_candidate_representations(self):
        return torch.cat(self.candidate_representations, dim=0) if self.candidate_representations else None

    def save_user_representations(self, path):
        if self.save_representations and self.user_representations:
            all_users = torch.cat(self.user_representations, dim=0)
            with open(path, 'wb') as f:
                pickle.dump(all_users.detach().cpu().numpy(), f)
            print(f"User representations saved to {path}")
        else:
            print("User representations saving is skipped or no user representations to save.")

    def save_candidate_representations(self, path):
        if self.save_representations and self.candidate_representations:
            all_candidates = torch.cat(self.candidate_representations, dim=0)
            with open(path, 'wb') as f:
                pickle.dump(all_candidates.detach().cpu().numpy(), f)
            print(f"Candidate representations saved to {path}")
        else:
            print("Candidate representations saving is skipped or no candidate representations to save.")


class BCELogitsRankingTrainerSimplified(RankingTrainerSimplified):
    def _init_loss(self):
        self.L = nn.functional.binary_cross_entropy_with_logits

    def forward(self, batch) -> torch.Tensor:
        batch_to_device(batch, self.device)
        return self.model.forward(batch)

    def _train_step(self, batch: dict) -> dict:
        results = super()._train_step(batch)
        results['logits'] = torch.sigmoid(results['logits'].clone().detach())
        return results

    @torch.no_grad()
    def _test_step(self, batch: dict) -> dict:
        results = super()._test_step(batch)
        results['scores'] = torch.sigmoid(torch.tensor(results['scores']))
        return results


class MSERankingTrainerSimplified(RankingTrainerSimplified):
    def _init_loss(self):
        def loss(prediction, target, weight: Optional[torch.tensor] = None):
            l = nn.functional.mse_loss(prediction, target, reduction='none') if weight is not None else nn.functional.mse_loss(prediction, target)
            return torch.mean(l * weight) if weight is not None else l
        self.L = loss

    def forward(self, batch) -> torch.Tensor:
        batch_to_device(batch, self.device)
        oupt, user_interests = self.model.forward(batch)
        return torch.relu(oupt)

    def _train_step(self, batch: dict) -> dict:
        return super()._train_step(batch)

    @torch.no_grad()
    def _test_step(self, batch: dict) -> dict:
        return super()._test_step(batch)
