# RecipeSnap - a lightweight pretrained model to predict recipe from image

We implemented a lightweight image to recipe prediction model. We used MobileNet_v2 as image encoder backbone. This model only has 3 million parameters and can be easily deployed on smartphone or other smart tablets.   

## Cloning 

```
git clone https://github.com/jianfa/RecipeSnap-a-lightweight-recipe-to-image-model.git
```

## Installation

- Create conda environment: ```conda env create -f environment.yml```
- Activate it with ```conda activate recipe_snap```


## Data preparation
Recipe embeddings can be download here:
1. feats_test.pkl (Included in data/recipe_embeddings folder)
2. [feats_train.pkl](https://drive.google.com/file/d/17UJyO00yRzwn5hnZ4-wMfH1vkMihyqNn/view?usp=sharing) (**Optional**)   

Please save these two files to data/recipe_embeddings/

Recipes can be download here:
1. test.pkl (Included in data/recipe_lib folder)
2. test_noimage.pkl (Included in data/recipe_lib folder)
3. [train.pkl](https://drive.google.com/file/d/17UJyO00yRzwn5hnZ4-wMfH1vkMihyqNn/view?usp=sharing) (**Optional**)
4. [train_noimages.pkl](https://drive.google.com/file/d/17UJyO00yRzwn5hnZ4-wMfH1vkMihyqNn/view?usp=sharing) (**Optional**)

Plase save these two files to data/recipe_lib/


## Training

We used pretrained recipe encoder from [image-to-recipe-transformers](https://github.com/amzn/image-to-recipe-transformers) and dataset from [Recipe1M](http://im2recipe.csail.mit.edu/dataset/download) to train our model. 

## Evaluation
| Split | loss | MedR | Recall_1 | Recall_5 | Recall_10 |  
|-------|------|------|----------|----------|-----------|  
| train | 0.0133 | 2.0000 | 0.4536 | 0.7911 | 0.8913 |  
| val  | 0.0267 | 2.0000 | 0.4123 | 0.7187 | 0.8210 |  

## Usage

Both demo.py and display.ipynb provide a usage example.

## License

This project is licensed under the Apache-2.0 License.