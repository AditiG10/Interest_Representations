from os import PathLike
import wandb as wb
import yaml
import os
from os.path import join, exists
import torch
import random
from os import PathLike
import argparse
from dotmap import DotMap

from nrs.models import make_model
from nrs.data.mind import make_mind_data
from nrs import training
import wandb
import numpy as np
from sklearn.model_selection import KFold
from os import PathLike
from nrs.models import make_model
from nrs.data.mind import make_mind_data
from nrs import training


def train_with_kfold(cfg, model, full_train_ds, test_ds, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=cfg.random_seed)
    
    all_fold_metrics = {
        'auc': [],
        'ndcg@5': [],
        'ndcg@10': [],
        'rr': [],
        'ctr@1': [],
        'acc': [],
        'rec': [],
        'prec': [],
        'train_loss': [],
        'test_loss': []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_ds)):
        print(f"Training fold {fold + 1}/{k}")
        
        train_idx = [int(idx) for idx in train_idx]
        val_idx = [int(idx) for idx in val_idx]
        
        train_subset = torch.utils.data.Subset(full_train_ds, train_idx)
        val_subset = torch.utils.data.Subset(full_train_ds, val_idx)
        
        # Use the BCELogitsRankingTrainerSimplified class and set save_representations to False
        trainer = training.BCELogitsRankingTrainerSimplified(
            cfg, model, train_subset, val_subset, disentanglement_loss_fn='cosine', save_representations=False
        )
        
        trainer.train()
        
        val_results = trainer._test_iteration()

        # Initialize fold_metrics with None for safety
        fold_metrics = {key: None for key in all_fold_metrics.keys()}

        # Calculate each metric safely
        fold_metrics['auc'] = np.mean([res.get('auc', 0) for res in val_results])
        fold_metrics['ndcg@5'] = np.mean([res.get('ndcg@5', 0) for res in val_results])
        fold_metrics['ndcg@10'] = np.mean([res.get('ndcg@10', 0) for res in val_results])
        fold_metrics['rr'] = np.mean([res.get('rr', 0) for res in val_results])
        fold_metrics['ctr@1'] = np.mean([res.get('ctr@1', 0) for res in val_results])
        fold_metrics['acc'] = np.mean([res.get('acc', 0) for res in val_results])
        fold_metrics['rec'] = np.mean([res.get('rec', 0) for res in val_results])
        fold_metrics['prec'] = np.mean([res.get('prec', 0) for res in val_results])
        fold_metrics['test_loss'] = np.mean([res.get('loss', 0) for res in val_results])

        # Append fold metrics to the aggregate metrics list
        for key in all_fold_metrics:
            all_fold_metrics[key].append(fold_metrics[key])

        # Ensure the train loss is recorded after training
        train_results = trainer._train_iteration()
        train_loss = np.mean([res['loss'] for res in train_results])
        all_fold_metrics['train_loss'].append(train_loss)

    # Final aggregation across all folds
    for key in all_fold_metrics:
        metric_values = [m for m in all_fold_metrics[key] if m is not None]
        avg_metric = np.mean(metric_values) if metric_values else 0
        std_metric = np.std(metric_values) if metric_values else 0
        print(f"{key}: {avg_metric:.4f} ± {std_metric:.4f}")

    return all_fold_metrics



def train(config_path: PathLike):
    print('init wandb')
    cfg_dict = yaml.full_load(open(config_path))
    dir = join(cfg_dict['dir'], cfg_dict['name'])
    if not exists(dir):
        os.makedirs(dir)

    wandb.init(
        config=cfg_dict,
        dir=dir,
        project=cfg_dict['project'],
        tags=cfg_dict['tags'],
        notes=cfg_dict['notes'],
        name=cfg_dict['name'],
        mode=cfg_dict['mode']
    )
    cfg = wandb.config

    torch.manual_seed(cfg.random_seed)
    random.seed(cfg.random_seed)

    print('init model')
    model = make_model(cfg)

    print('preparing data')
    full_train_ds, test_ds = make_mind_data(cfg)

    metrics = train_with_kfold(cfg, model, full_train_ds, test_ds, k=5)
    
    print('k-fold cross-validation completed')
    print('Final aggregated results:')
    for key in metrics:
        avg_metric = np.mean(metrics[key])
        std_metric = np.std(metrics[key])
        wandb.log({f'final_{key}': avg_metric, f'final_{key}_std': std_metric})
        print(f"{key}: {avg_metric:.4f} ± {std_metric:.4f}")

    print('Re-training on the full training data and evaluating on the MINDlarge_dev dataset')
    final_trainer = training.BCELogitsRankingTrainer(cfg, model, full_train_ds, test_ds)
    final_trainer.train()
    final_results = final_trainer._test_iteration()

    wandb.log({
        'final_test_auc': final_results['auc'],
        'final_test_ndcg@5': final_results['ndcg@5'],
        'final_test_ndcg@10': final_results['ndcg@10'],
        'final_test_mrr': final_results['rr'],
        'final_test_ctr@1': final_results['ctr@1'],
        'final_test_acc': final_results['acc'],
        'final_test_rec': final_results['rec'],
        'final_test_prec': final_results['prec'],
        'final_test_loss': final_results['loss'],
    })

    print('Final test results on MINDlarge_dev:', final_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        help='path to config file',
                        default='config/mind_standard2.yml'
                        )
    args = parser.parse_args()
    train(config_path=args.config)
