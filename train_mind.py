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


def train(config_path: PathLike):

    print('init wandb')
    cfg_dict = yaml.full_load(open(config_path))
    dir = join(cfg_dict['dir'], cfg_dict['name'])
    if not exists(dir):
        os.makedirs(dir)

    # NOTE: weights&biases (wandb) is a logging tool that I use
    # if everything else runs you can check it out and create an account,
    # for now I disabled it.
    wb.init(
        config=cfg_dict,
        dir=dir,
        project=cfg_dict['project'],
        tags=cfg_dict['tags'],
        notes=cfg_dict['notes'],
        name=cfg_dict['name'],
        mode=cfg_dict['mode']
    )
    cfg = wb.config

    # cfg = DotMap(cfg_dict)

    torch.manual_seed(cfg.random_seed)
    random.seed(cfg.random_seed)

    print('init model')
    model = make_model(cfg)
    # wb.watch(model)

    print('preparing data')
    train_ds, test_ds = make_mind_data(cfg)

    '''
    trainer = training.MSERankingTrainer(
        cfg, model, train_ds, test_ds
    )
    '''

    trainer = training.BCELogitsRankingTrainer(
        cfg, model, train_ds, test_ds
    )
    
    print('starting training')
    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        help='path to config file',
                        default='config/mind_standard.yml'
                        )
    args = parser.parse_args()
    train(config_path=args.config)
