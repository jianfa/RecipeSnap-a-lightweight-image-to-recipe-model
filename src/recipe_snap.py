# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from sklearn.metrics import pairwise_distances
from collections import defaultdict
# import torch.nn.functional as F
from recipe_encoder import * 
from image_encoder import *
from utils.preprocessing import get_image_loader 
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
        self.image_encoder = ImageEmbedding(output_size=output_size, image_model=image_model)
        self.checkpoint_dir = checkpoint_dir
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

    def compute_image_embedding(self, loader):
        image_num = len(loader)
        loader = iter(loader)
        img_embeddings = []
        img_names = []
        for _ in range(image_num):
            img, img_name = loader.next()
            emb = self.image_encoder(img)
            img_names.append(img_name)
            img_embeddings.append(emb.cpu().detach().numpy())
        img_embeddings = np.vstack(img_embeddings)
        return img_embeddings, img_names
        

    def load_image(self, image_dir, batch_size=1, resize=256, im_size=224, augment=True, mode='predict',drop_last=True):
        loader, dataset = get_image_loader(image_dir, resize=resize, im_size=im_size, batch_size=batch_size, 
                                                                augment=augment, mode=mode,drop_last=drop_last)
        print(f"{len(loader)} image loaded")
        return loader, dataset

    def load_recipe(self, recipe_emb_path="../data/recipe_embeddings/recipe_embeddings_feats_test.pkl", 
                    recipe_lib_dir="../data/recipe_lib", split="test"):
        with open(recipe_emb_path, 'rb') as f:
            self.recipe_embs = pickle.load(f)
            self.recipe_ids = pickle.load(f)
        print(f"Succeed to load recipe embedding from ... {recipe_emb_path}")
        print(f"Recipe embedding shape: {self.recipe_embs.shape}")

        file_path = os.path.join(recipe_lib_dir, split+'.pkl')
        with open(file_path, 'rb') as f:
            self.recipe_lib = pickle.load(f)
        with open(os.path.join(recipe_lib_dir, split + "_noimages.pkl"), 'rb') as f:
            self.recipe_lib.update(pickle.load(f))
        print(f"Succeed to load recipe dictionary from ... {recipe_lib_dir}")
        print(f"Recipe library size {len(self.recipe_lib)}")


    def predict(self, image_dir, max_k=5):
        loader, dataset = self.load_image(image_dir)
        img_embs, img_names = self.compute_image_embedding(loader)
        dists = pairwise_distances(img_embs, self.recipe_embs, metric='cosine')
        retrieved_idxs_recs = np.argpartition(dists, range(max_k), axis=-1)[:,:max_k]
        retrieved_recipes_dict = defaultdict(list)
        for i, img_name in enumerate(img_names):
            for rec_id in retrieved_idxs_recs[i]:
                retrieved_recipes_dict[img_name].append(self.recipe_lib[self.recipe_ids[rec_id]])

        return retrieved_recipes_dict

    def update_recipe_dict(self, new_recipes):
        None 
