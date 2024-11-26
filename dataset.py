import kagglehub
import pandas as pd

PATH = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")

def getRawRecipes(path=PATH):
    RAW_recipes=pd.read_csv(path+'/RAW_recipes.csv')
    return RAW_recipes.dropna()
def getRawInteractions(path=PATH):
    RAW_interactions=pd.read_csv(path+'/RAW_interactions.csv')
    return RAW_interactions.dropna()