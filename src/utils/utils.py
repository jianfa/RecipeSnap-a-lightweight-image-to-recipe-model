# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Utils functions
"""
import os
import torch
import nltk




def load_checkpoint(checkpoints_dir, map_loc, suff='best'):
    model_state_dict = torch.load(os.path.join(checkpoints_dir,'model-'+suff+'.ckpt'),map_location=map_loc)
    return model_state_dict

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