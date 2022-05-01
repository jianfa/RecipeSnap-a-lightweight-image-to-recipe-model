# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances
from collections import defaultdict
# import torch.nn.functional as F
# from recipe_encoder import * 
# from image_encoder import *
# from utils.preprocessing import get_image_loader 
# from utils.utils import load_checkpoint, count_parameters

ROOT = '/Users/jfachen/Desktop/Final Project/recipe_snap/src/'
IMAGE_DIR = ROOT + '../images'
CHECKPOINT_DIR = ROOT + "../checkpoints/model"
RECIPE_EMB_PATH = ROOT + "../data/recipe_embeddings/recipe_embeddings_feats_test.pkl" 
RECIPE_DICT_PATH = ROOT + "../data/traindata/test.pkl"
RECIPE_DICT_NOIMAGES_PATH = ROOT + "../data/traindata/test_noimages.pkl"

# Initialize
image_encoder = ImageEmbedding(output_size=1024, image_model='mobilenet_v2')
checkpoint_dir = CHECKPOINT_DIR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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

# def load_image(self, image_dir, batch_size=1, resize=256, im_size=224, augment=True, mode='predict',drop_last=True):
loader, dataset = get_image_loader(IMAGE_DIR, resize=256, im_size=224, batch_size=1, 
                                                        augment=False, mode='predict',drop_last=False)
print(f"{len(loader)} image loaded")

# img = Image.open(os.path.join(IMAGE_DIR, dataset.image_names[0]))
# def compute_image_embedding(self, loader):
image_num = len(loader)
loader = iter(loader)
img_embs = []
image_names = []
for _ in range(image_num):
    img, img_name = loader.next()
    emb =  image_encoder(img)
    image_names.append(img_name)
    img_embs.append(emb.cpu().detach().numpy())
img_embs = np.vstack(img_embs)

        



# def load_recipe(self, recipe_emb_path="../data/recipe_embeddings/recipe_embeddings_feats_test.pkl", recipe_dict_path="../data/traindata/test.pkl"):
with open(RECIPE_EMB_PATH, 'rb') as f:
        recipe_embs = pickle.load(f)
        recipe_ids = pickle.load(f)
print(f"Succeed to load recipe embedding from ... {RECIPE_EMB_PATH}")
print(f"Recipe embedding shape: {recipe_embs.shape}")
with open(RECIPE_DICT_PATH, 'rb') as f:
        recipe_dict = pickle.load(f)

with open(RECIPE_DICT_NOIMAGES_PATH, 'rb') as f:
        recipe_dict.update(pickle.load(f))
print(f"Succeed to load recipe library from ... {RECIPE_DICT_PATH}")
print(f"Recipe library size {len(recipe_dict)}")

# def predict(self, image_dir, max_k=5):
# loader, dataset =  load_image(IMAGE_DIR)
# img_embs, image_names =  compute_image_embedding(loader)
max_k = 5
dists = pairwise_distances(img_embs,  recipe_embs, metric='cosine')
retrieved_idxs_recs = np.argpartition(dists, range(max_k), axis=-1)[:,:max_k]
retrieved_recipes_dict = defaultdict(list)
for i, img_name in enumerate(image_names):
    for rec_id in retrieved_idxs_recs[i]:
        if recipe_ids[rec_id] not in recipe_dict:
            print(recipe_ids[rec_id])
            continue
        retrieved_recipes_dict[img_name].append( recipe_dict[ recipe_ids[rec_id]])

img_idx = 0 
for img_idx in range(3):
    print(f"Image is {image_names[img_idx]}")
    for rank in range(5):
        print(f"Rank {rank}: {retrieved_recipes_dict[image_names[img_idx]][rank]['title']}")
    


