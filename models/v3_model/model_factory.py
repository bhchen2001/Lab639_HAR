import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_uniform_
import torch
import torchvision.models.video as video_models
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
from models.v3_model.v3_backbone import VideoTransformer as V3


def cosine_pairwise_dist(x, y):
    assert x.shape[1] == y.shape[1], "both sets of features must have same shape"
    return nn.functional.cosine_similarity(x[:, None, :], y[None, :, :], dim=-1)  

def weights_init(m):
    if isinstance(m, nn.Linear):
        kaiming_uniform_(m.weight.data)

def build_model(version, num_views, num_actions):
    if version == 'v3':
        model = V3(num_views, num_actions)
    model.apply(weights_init)
    return model