import torch
from os import PathLike
from dotmap import DotMap

from .make_model import make_model


def load_model_from_ckpt(
    path: PathLike
):
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    cfg = DotMap(ckpt['config'])
    model = make_model(cfg)
    model.load_state_dict(ckpt['state_dict'])
    return model, cfg