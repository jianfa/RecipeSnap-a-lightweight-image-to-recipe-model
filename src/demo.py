from recipe_snap import * 
from PIL import ImageOps, Image
from torchvision import transforms
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')

# image resizing for display
tfs = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256)])

def plot_image(image_path):
    img = tfs(Image.open(image_path).convert("RGB"))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def print_recipe(recipe, rank):
    print(f"\n Recipe likelihood rank: {rank}")
    print(f"Title: {recipe['title']}")
    print("Ingredient: ")
    print("\t" + "\n \t".join(recipe['ingredients']))
    print("Instruction: ")
    print("\t" + "\n \t".join(recipe['instructions']))

def main():
    # Update with your paths and model names, the following are default values
    image_dir = '../images'
    checkpoint_dir = "../checkpoints/model"
    recipe_emb_path = "../data/recipe_embeddings/recipe_embeddings_feats_test.pkl" 
    recipe_dict_dir = "../data/recipe_dict"
    new_recipe_emb_path="../data/recipe_embeddings/new_recipe_embeddings.pkl"
    new_recipe_dict_path="../data/recipe_dict/new_recipe.pkl"
    
    new_recipes = [{"title":"Pan-Fried Chinese Pancakes",
                    "ingredients":["1/4 teaspoon salt", "3/4 cup warm water"], 
                    "instructions":["Dissolve salt in warm water, and mix in 1 cup of flour to make a soft dough. Turn the dough out onto a well-floured work surface, and knead until slightly springy, about 5 minutes. If the dough is sticky, knead in 1/4 teaspoon of vegetable oil. Divide the dough into 8 equal-size pieces, and keep the pieces covered with a cloth",
                    "In a bowl, mix 1/4 cup of flour with 1 tablespoon vegetable oil to make a mixture like fine crumbs."]}
                    ]

    rs = RecipeSnap(checkpoint_dir=checkpoint_dir)
    rs.load_image_encoder()
    rs.load_recipe(recipe_emb_path=recipe_emb_path, recipe_dict_dir=recipe_dict_dir)
    results = rs.predict(image_dir=image_dir)

    for img, recipes in results.items():
        # plot_image(os.path.join(image_dir, img[0]))
        print(f"Image {img}")
        for i, recipe in enumerate(recipes):
            print_recipe(recipe, i)

    rs.load_recipe_encoder()
    rs.update_recipe_lib(new_recipes=new_recipes)
    rs.save_recipe_lib(new_recipe_emb_path, new_recipe_dict_path)



if __name__=="__main__": 
    main()