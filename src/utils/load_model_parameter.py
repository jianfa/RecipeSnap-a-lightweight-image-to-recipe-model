# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Training script
"""
from inspect import Parameter
import os
import random
from collections import defaultdict
import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.cuda.amp import GradScaler, autocast
import gc
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from utils.loss import TripletLoss
from dataset import get_loader
from config import get_args
from models import get_model
from eval import computeAverageMetrics

MAP_LOC = None if torch.cuda.is_available() else 'cpu'

def load_checkpoint(checkpoints_dir, map_loc, suff='best'):

    assert os.path.exists(os.path.join(checkpoints_dir, 'args.pkl'))
    model_state_dict = torch.load(os.path.join(checkpoints_dir,
                                               'model-'+suff+'.ckpt'),
                                  map_location=map_loc)

    args = pickle.load(open(os.path.join(checkpoints_dir, 'args.pkl'), 'rb'))

    return getattr(args, 'batch_size'), model_state_dict


def load_recipe_dict(recipe_embedding_path):
    None 

def update_recipe_dict(new_recipe, recipe_dict):
    None 

def predict_top_k_recipes(recipe_dict, k=5):
    None 




def load_model_weight(model, model_state_dict):

    batch_size, model_dict = load_checkpoint(checkpoint_dir, 'best', MAP_LOC)
    # batch_size should be same as loaded
    model.load_state_dict(model_dict, strict=False)
    
     