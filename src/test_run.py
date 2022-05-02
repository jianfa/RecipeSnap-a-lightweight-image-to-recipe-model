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


image_dir = '../images'
checkpoint_dir = "../checkpoints/model"
recipe_emb_path = "../data/recipe_embeddings/recipe_embeddings_feats_test.pkl" 
recipe_lib_path = "../data/recipe_lib/test.pkl"
output_size=1024
image_model='mobilenet_v2'
batch_size=1
resize=256 
im_size=224 
augment=True, 
mode='predict',
drop_last=True
max_k=5
new_recipe_dict = [{"title":"Pan-Fried Chinese Pancakes",
                    "ingredients":["1/4 teaspoon salt", "3/4 cup warm water"], 
                    "instructions":["Dissolve salt in warm water, and mix in 1 cup of flour to make a soft dough. Turn the dough out onto a well-floured work surface, and knead until slightly springy, about 5 minutes. If the dough is sticky, knead in 1/4 teaspoon of vegetable oil. Divide the dough into 8 equal-size pieces, and keep the pieces covered with a cloth",
                    "In a bowl, mix 1/4 cup of flour with 1 tablespoon vegetable oil to make a mixture like fine crumbs."]}
                    ]

    # def __init__(self, checkpoint_dir='../checkpoints/model', output_size=1024, image_model='mobilenet_v2'):
        # checkpoint_dir=checkpoint_dir
        # output_size = output_size
        image_encoder = ImageEmbedding(output_size=output_size, image_model=image_model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # def load_image_encoder(self):
        print(f"Loading checkpoint from ... {checkpoint_dir}")
        map_loc = None if torch.cuda.is_available() else 'cpu'
        model_dict = load_checkpoint(checkpoint_dir,map_loc=map_loc,suff='best')
        image_encoder.load_state_dict(model_dict, strict=False)
        print("Loading checkpoint succeed.")
        print("image encoder", count_parameters(image_encoder))

        if device != 'cpu' and torch.cuda.device_count() > 1:
            image_encoder = torch.nn.DataParallel(image_encoder)
        image_encoder.to(device)
        image_encoder.eval()

    # def load_recipe_encoder(self):

        print(f"Loading recipe encoder ...")
        recipe_encoder = JointEmbedding(output_size=output_size, vocab_size=16303)
        map_loc = None if torch.cuda.is_available() else 'cpu'
        model_dict = load_checkpoint(checkpoint_dir,map_loc=map_loc,suff='best')
        recipe_encoder.load_state_dict(model_dict, strict=False)
        print("Loading recipe encoder succeed.")
        print("image encoder", count_parameters(recipe_encoder))

        if device != 'cpu' and torch.cuda.device_count() > 1:
            recipe_encoder = torch.nn.DataParallel(recipe_encoder)
        recipe_encoder.to(device)
        recipe_encoder.eval()

    # def compute_image_embedding(self, loader):
        loader, dataset = get_image_loader(image_dir, resize=resize, im_size=im_size, batch_size=batch_size, 
                                                                augment=augment, mode=mode,drop_last=drop_last)
        print(f"{len(loader)} image loaded")
        image_num = len(loader)
        loader = iter(loader)
        img_embeddings = []
        img_names = []
        for _ in range(image_num):
            img, img_name = loader.next()
            img = img.to(device)
            with torch.no_grad():
                emb = image_encoder(img)
            img_names.extend(img_name)
            img_embeddings.append(emb.cpu().detach().numpy())
        img_embeddings = np.vstack(img_embeddings)
        # return img_embeddings, img_names

    # def compute_recipe_embedding(self, loader):
        recipe_path = None
        recipe_dict = new_recipe_dict 
        loader, dataset = get_recipe_loader(recipe_path=recipe_path, recipe_dict=recipe_dict, batch_size=batch_size, 
                                                drop_last=drop_last)
        recipe_num = len(loader)
        loader = iter(loader)
        recipe_embeddings = []
        recipe_ids = []
        for _ in range(recipe_num):
            title, ingrs, instrs, ids = loader.next()
            title = title.to(device)
            ingrs = ingrs.to(device)
            instrs = instrs.to(device)
            with torch.no_grad():
                recipe_emb = recipe_encoder(title, ingrs, instrs)
            recipe_ids.extend(ids)
            recipe_embeddings.append(recipe_emb.cpu().detach().numpy())
        recipe_embeddings = np.vstack(recipe_embeddings)
        return recipe_embeddings, recipe_ids
        

    # def load_image(self, image_dir, batch_size=1, resize=256, im_size=224, augment=True, mode='predict',drop_last=True):


        return loader, dataset

    # def load_recipe(self, recipe_path=None, recipe_dict=None, batch_size=1,drop_last=True):

        print(f"{len(loader)} recipe loaded")
        return loader, dataset

    # def load_recipe_lib(self, recipe_emb_path="../data/recipe_embeddings/recipe_embeddings_feats_test.pkl", 
    #                 recipe_lib_path="../data/recipe_lib/test.pkl"):
        with open(recipe_emb_path, 'rb') as f:
            recipe_embs = pickle.load(f)
            recipe_ids = pickle.load(f)
        print(f"Succeed to load recipe embedding from ... {recipe_emb_path}")
        print(f"Recipe embedding shape: {recipe_embs.shape}")

        with open(recipe_lib_path, 'rb') as f:
            recipe_lib = pickle.load(f)
        noimage_file_path = recipe_lib_path[:-4] + "_noimages.pkl"
        if os.path.exists(noimage_file_path):
            with open(noimage_file_path, 'rb') as f:
                recipe_lib.update(pickle.load(f))
        print(f"Succeed to load recipe library from ... {recipe_lib_path}")
        print(f"Recipe library size {len(recipe_lib)}")


    # def predict(self, image_dir, max_k=5):
    #     loader, dataset = load_image(image_dir)
    #     img_embs, img_names = compute_image_embedding(loader)
        img_embs = img_embeddings
        dists = pairwise_distances(img_embs, recipe_embs, metric='cosine')
        retrieved_idxs_recs = np.argpartition(dists, range(max_k), axis=-1)[:,:max_k]
        retrieved_recipes_dict = defaultdict(list)
        for i, img_name in enumerate(img_names):
            for rec_id in retrieved_idxs_recs[i]:
                retrieved_recipes_dict[img_name].append(recipe_lib[recipe_ids[rec_id]])

        return retrieved_recipes_dict

    # def update_recipe_lib(self, new_recipe_dict):
        print("Updating recipe lib ...")
        print(f"Before update, there are {len(recipe_lib)} recipes in library")
        new_recipe_dict = recipe_preprocessing(new_recipe_dict)
        loader, dataset = load_recipe(new_recipe_dict)
        new_recipe_embs, new_recipe_ids = compute_recipe_embedding(loader)
        new_recipe_embs = recipe_embeddings
        recipe_embs = np.concatenate((recipe_embs, new_recipe_embs))
        recipe_ids.extend(new_recipe_ids)
        recipe_lib.update(new_recipe_dict)
        print(f"After update, there are {len(recipe_lib)} recipes in library")


    def save_recipe_lib(self, new_recipe_emb_path, new_recipe_lib_path):
        with open(new_recipe_emb_path, 'wb') as f:
            pickle.dump(recipe_embs, f)
            pickle.dump(recipe_ids, f)

        with open(new_recipe_lib_path, 'wb') as f:
            pickle.dump(recipe_lib, f)
        
