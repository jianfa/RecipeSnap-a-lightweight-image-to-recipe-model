from recipe_snap import * 
from PIL import ImageOps, Image
from torchvision import transforms
import matplotlib.pyplot as plt

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
    recipe_lib_dir = "../data/recipe_lib"

    rs = RecipeSnap(checkpoint_dir=checkpoint_dir)
    rs.load_image_encoder()
    rs.load_recipe(recipe_emb_path=recipe_emb_path, recipe_lib_dir=recipe_lib_dir)
    results = rs.predict(image_dir=image_dir)

    for img, recipes in results.items():
        # plot_image(os.path.join(image_dir, img[0]))
        print(f"Image {img}")
        for i, recipe in enumerate(recipes):
            print_recipe(recipe, i)





if __name__=="__main__": 
    main()