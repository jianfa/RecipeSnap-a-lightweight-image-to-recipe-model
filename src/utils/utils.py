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
# from utils.loss import TripletLoss
# from dataset import get_loader
# from config import get_args
# from models import get_model
# from eval import computeAverageMetrics



def load_checkpoint(checkpoints_dir, map_loc, suff='best'):
    model_state_dict = torch.load(os.path.join(checkpoints_dir,'model-'+suff+'.ckpt'),map_location=map_loc)
    return model_state_dict


def load_recipe_dict(recipe_embedding_path):
    None 

def update_recipe_dict(new_recipe, recipe_dict):
    None 

def predict_top_k_recipes(recipe_dict, k=5):
    None 



# def load_model_weight(model, model_state_dict):

#     batch_size, model_dict = load_checkpoint(checkpoint_dir, 'best', MAP_LOC)
#     # batch_size should be same as loaded
#     model.load_state_dict(model_dict, strict=False)
    


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_token_ids(sentence, vocab):
    """
    get vocabulary tokens for each word in a sentence
    """
    tok_ids = []
    tokens = nltk.tokenize.word_tokenize(sentence.lower())

    tok_ids.append(vocab['<start>'])
    for token in tokens:
        if token in vocab:
            tok_ids.append(vocab[token])
        else:
            # unk words will be ignored
            tok_ids.append(vocab['<unk>'])
    tok_ids.append(vocab['<end>'])
    return tok_ids

def list2Tensors(input_list):
    """Given a list of lists of variable-length elements, return a 2D tensor padded with 0s
    """
    max_seq_len = max(map(len, input_list))
    output = [v + [0] * (max_seq_len - len(v)) for v in input_list]

    return torch.Tensor(output)