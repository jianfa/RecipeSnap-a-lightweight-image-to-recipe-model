# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from recipe_encoder import * 
from image_encoder import *
from utils.preprocessing import get_image_loader, get_recipe_loader, recipe_preprocessing 
from utils.utils import load_checkpoint, count_parameters



class RecipeSnap(object):
    """ a light-weight pretrained model to predict recipe from image

    Parameters
    ----------
    recipe_dict : str
        Path of recipe dictionary file.
    checkpoint_dir : str
        Path of checkpoint folder.
    """
    def __init__(self, checkpoint_dir='../checkpoints/model', output_size=1024, image_model='mobilenet_v2'):
        self.checkpoint_dir=checkpoint_dir
        self.output_size = output_size
        self.image_encoder = ImageEmbedding(output_size=output_size, image_model=image_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_image_encoder(self):
        print(f"Loading checkpoint from ... {self.checkpoint_dir}")
        map_loc = None if torch.cuda.is_available() else 'cpu'
        model_dict = load_checkpoint(self.checkpoint_dir,map_loc=map_loc,suff='best')
        self.image_encoder.load_state_dict(model_dict, strict=False)
        print("Loading checkpoint succeed.")
        print("image encoder", count_parameters(self.image_encoder))

        if self.device != 'cpu' and torch.cuda.device_count() > 1:
            self.image_encoder = torch.nn.DataParallel(self.image_encoder)
        self.image_encoder.to(self.device)
        self.image_encoder.eval()

    def load_recipe_encoder(self):
        print(f"Loading recipe encoder ...")
        self.recipe_encoder = JointEmbedding(output_size=self.output_size, vocab_size=16303)
        map_loc = None if torch.cuda.is_available() else 'cpu'
        model_dict = load_checkpoint(self.checkpoint_dir,map_loc=map_loc,suff='best')
        self.recipe_encoder.load_state_dict(model_dict, strict=False)
        print("Loading recipe encoder succeed.")
        print("image encoder", count_parameters(self.recipe_encoder))

        if self.device != 'cpu' and torch.cuda.device_count() > 1:
            self.recipe_encoder = torch.nn.DataParallel(self.recipe_encoder)
        self.recipe_encoder.to(self.device)
        self.recipe_encoder.eval()

    def compute_image_embedding(self, loader):
        image_num = len(loader)
        loader = iter(loader)
        img_embeddings = []
        img_names = []
        for _ in range(image_num):
            img, img_name = loader.next()
            img = img.to(self.device)
            with torch.no_grad():
                emb = self.image_encoder(img)
            img_names.extend(img_name)
            img_embeddings.append(emb.cpu().detach().numpy())
        img_embeddings = np.vstack(img_embeddings)
        return img_embeddings, img_names

    def compute_recipe_embedding(self, loader):
        recipe_num = len(loader)
        loader = iter(loader)
        recipe_embeddings = []
        recipe_ids = []
        for _ in range(recipe_num):
            title, ingrs, instrs, ids = loader.next()
            title = title.to(self.device)
            ingrs = ingrs.to(self.device)
            instrs = instrs.to(self.device)
            with torch.no_grad():
                recipe_emb = self.recipe_encoder(title, ingrs, instrs)
            recipe_ids.extend(ids)
            recipe_embeddings.append(recipe_emb.cpu().detach().numpy())
        recipe_embeddings = np.vstack(recipe_embeddings)
        return recipe_embeddings, recipe_ids
        

    def load_image(self, image_dir, batch_size=1, resize=256, im_size=224, augment=True, mode='predict',drop_last=True):
        loader, dataset = get_image_loader(image_dir, resize=resize, im_size=im_size, batch_size=batch_size, 
                                                                augment=augment, mode=mode,drop_last=drop_last)
        print(f"{len(loader)} image loaded")
        return loader, dataset

    def load_recipe(self, recipe_path=None, recipe_dict=None, batch_size=1,drop_last=True):
        loader, dataset = get_recipe_loader(recipe_path=recipe_path, recipe_dict=recipe_dict, batch_size=batch_size, 
                                                drop_last=drop_last)
        print(f"{len(loader)} recipe loaded")
        return loader, dataset

    def load_recipe_lib(self, recipe_emb_path="../data/recipe_embeddings/recipe_embeddings_feats_test.pkl", 
                    recipe_dict_path="../data/recipe_dict/test.pkl"):
        with open(recipe_emb_path, 'rb') as f:
            self.recipe_embs = pickle.load(f)
            self.recipe_ids = pickle.load(f)
        print(f"Succeed to load recipe embedding from ... {recipe_emb_path}")
        print(f"Recipe embedding shape: {self.recipe_embs.shape}")

        with open(recipe_dict_path, 'rb') as f:
            self.recipe_dict = pickle.load(f)
        noimage_file_path = recipe_dict_path[:-4] + "_noimages.pkl"
        if os.path.exists(noimage_file_path):
            with open(noimage_file_path, 'rb') as f:
                self.recipe_dict.update(pickle.load(f))
        print(f"Succeed to load recipe library from ... {recipe_dict_path}")
        print(f"Recipe library size {len(self.recipe_dict)}")


    def predict(self, image_dir, max_k=5):
        loader, dataset = self.load_image(image_dir)
        img_embs, img_names = self.compute_image_embedding(loader)
        dists = pairwise_distances(img_embs, self.recipe_embs, metric='cosine')
        retrieved_idxs_recs = np.argpartition(dists, range(max_k), axis=-1)[:,:max_k]
        retrieved_recipes_dict = defaultdict(list)
        for i, img_name in enumerate(img_names):
            for rec_id in retrieved_idxs_recs[i]:
                retrieved_recipes_dict[img_name].append(self.recipe_dict[self.recipe_ids[rec_id]])

        return retrieved_recipes_dict

    def update_recipe_lib(self, new_recipes):
        print("Updating recipe lib ...")
        print(f"Before update, there are {len(self.recipe_dict)} recipes in library")
        new_recipe_dict = recipe_preprocessing(new_recipes)
        loader, dataset = self.load_recipe(recipe_dict=new_recipe_dict)
        new_recipe_embs, new_recipe_ids = self.compute_recipe_embedding(loader)
        self.recipe_embs = np.concatenate((self.recipe_embs, new_recipe_embs))
        self.recipe_ids.extend(new_recipe_ids)
        self.recipe_dict.update(new_recipe_dict)
        print(f"After update, there are {len(self.recipe_dict)} recipes in library")


    def save_recipe_lib(self, new_recipe_emb_path, new_recipe_dict_path):
        with open(new_recipe_emb_path, 'wb') as f:
            pickle.dump(self.recipe_embs, f)
            pickle.dump(self.recipe_ids, f)

        with open(new_recipe_dict_path, 'wb') as f:
            pickle.dump(self.recipe_dict, f)
        
