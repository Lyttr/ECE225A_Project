import pandas as pd
import dataset
from wordcloud import WordCloud
import matplotlib.pyplot as plt
RawRecipe=dataset.getRawRecipes()

print(RawRecipe.describe()['minutes'],RawRecipe.describe()['n_steps'],RawRecipe.describe()['n_ingredients'])
