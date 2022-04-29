# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Dataset and dataloader functions
"""

import os
import json
import random
random.seed(1234)
from random import choice
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
# from utils.utils import get_token_ids, list2Tensors
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import multiprocessing
import nltk


def text_preprocessing(new_recipe_dict):
    None 

def img_preprocessing(new_imgs):
    None 

def get_img_loader():
    None 

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


def pad_input(input):
    """
    creates a padded tensor to fit the longest sequence in the batch
    """
    if len(input[0].size()) == 1:
        l = [len(elem) for elem in input]
        targets = torch.zeros(len(input), max(l)).long()
        for i, elem in enumerate(input):
            end = l[i]
            targets[i, :end] = elem[:end]
    else:
        n, l = [], []
        for elem in input:
            n.append(elem.size(0))
            l.append(elem.size(1))
        targets = torch.zeros(len(input), max(n), max(l)).long()
        for i, elem in enumerate(input):
            targets[i, :n[i], :l[i]] = elem
    return targets


def collate_fn_image(data):
    """ collate to consume and batchify image data
    """
    image, image_names = zip(*data)
    if image[0] is not None:
        # Merge images (from tuple of 3D tensor to 4D tensor).
        image = torch.stack(image, 0)
    else:
        image = None
    return image, image_names

def collate_fn_recipe(data):
    """ collate to consume and batchify recipe data
    """

    # Sort a data list by caption length (descending order).
    titles, ingrs, instrs, ids = zip(*data)
    title_targets = pad_input(titles)
    ingredient_targets = pad_input(ingrs)
    instruction_targets = pad_input(instrs)

    return title_targets, ingredient_targets, instruction_targets, ids


class Recipe(Dataset):
    """Recipe Dataset class

    Parameters
    ----------
    new_recipe_path : string
        Path to new recipe file, must in pickle format.
    max_ingrs : int
        Maximum number of ingredients to use.
    max_instrs : int
        Maximum number of instructions to use.
    max_length_ingrs : int
        Maximum length of ingredient sentences.
    max_length_instrs : int
        Maximum length of instruction sentences.
    """

    def __init__(self, new_recipe_path,
                 max_ingrs=20,
                 max_instrs=20,
                 max_length_ingrs=15,
                 max_length_instrs=15):

        #load vocabulary
        self.vocab_inv = pickle.load(open('../data/vocab.pkl', 'rb'))
        self.vocab = {}
        for k, v in self.vocab_inv.items():
            if type(v) != str:
                v = v[0]
            self.vocab[v] = k

        self.data = pickle.load(open(new_recipe_path,'rb'))
        self.ids = list(self.data.keys())

        self.max_ingrs = max_ingrs
        self.max_instrs = max_instrs
        self.max_length_ingrs = max_length_ingrs
        self.max_length_instrs = max_length_instrs

    def __getitem__(self, idx):

        entry = self.data[self.ids[idx]]
        title = entry['title']
        ingrs = entry['ingredients']
        instrs = entry['instructions']

        # turn text into indexes
        title = torch.Tensor(get_token_ids(title, self.vocab)[:self.max_length_instrs])
        instrs = list2Tensors([get_token_ids(instr, self.vocab)[:self.max_length_instrs] for instr in instrs[:self.max_instrs]])
        ingrs = list2Tensors([get_token_ids(ingr, self.vocab)[:self.max_length_ingrs] for ingr in ingrs[:self.max_ingrs]])

        return title, ingrs, instrs, self.ids[idx]

    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        return self.ids

    def get_vocab(self):
        try:
            return self.vocab_inv
        except:
            return None

def get_recipe_loader(recipe_dir, batch_size,
               split='train', mode='train',
               drop_last=True,
               text_only_data=False):
    """Function to get dataset and dataloader for a data split

    Parameters
    ----------
    recipe_dir : string
        Path to Recipe1M dataset.
    batch_size : int
        Batch size.
    split : string
        Dataset split (train, val, or test)
    mode : string
        Loading mode (impacts augmentations & random sampling)
    drop_last : bool
        Whether to drop the last batch of data.
    text_only_data : type
        Whether to load text-only or paired samples.

    Returns
    -------
    loader : a pytorch DataLoader
    ds : a pytorch Dataset

    """

    ds = Recipe(recipe_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=multiprocessing.cpu_count(),
                        collate_fn=collate_fn_recipe, drop_last=drop_last)

    return loader, ds

class ImageDataset(Dataset):
    """Dataset class for Images

    Parameters
    ----------
    image_dir : string
        Path to images.
    transform : (callable, optional)
        A function/transform that takes in a PIL image and returns a transformed version.
    """

    def __init__(self, image_dir, transform=None):                           
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = [f for f in os.listdir(self.image_dir) \
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        img = Image.open(os.path.join(self.image_dir, image_name))
        if img is None:
            print(f"Fail to read image {image_name}")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = None
        return img, image_name

    def __len__(self):
        return len(self.image_names)

    # def get_ids(self):
    #     return self.image_names
        # return list(range(len(self.image_names)))

def get_image_loader(image_dir, resize=256, im_size=224, batch_size=1, augment=True, mode='predict',drop_last=True):
    """Function to get dataset and dataloader for images

    Parameters
    ----------
    image_dir : string
        Path to image folder.
    batch_size : int
        Batch size.
    resize : int
        Image size for resizing (keeps aspect ratio)
    im_size : int
        Image size for cropping.
    augment : bool
        Description of parameter `augment`.
    mode : string
        Loading mode (impacts augmentations & random sampling)
    drop_last : bool
        Whether to drop the last batch of data.

    Returns
    -------
    loader : a pytorch DataLoader
    ds : a pytorch Dataset

    """

    transforms_list = [transforms.Resize((resize))]

    if mode == 'train' and augment:
        # Image preprocessing, normalization for pretrained resnet
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.RandomCrop(im_size))

    else:
        transforms_list.append(transforms.CenterCrop(im_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)))

    transforms_ = transforms.Compose(transforms_list)

    ds = ImageDataset(image_dir, transform=transforms_)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=multiprocessing.cpu_count(),
                        # num_workers=1,
                        collate_fn=collate_fn_image, drop_last=drop_last)

    return loader, ds