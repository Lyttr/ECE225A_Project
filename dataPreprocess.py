import pandas as pd
import dataset
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
RawRecipe=dataset.getRawRecipes()
def numDataAnalyRecipe():
    print(RawRecipe.describe()['minutes'],RawRecipe.describe()['n_steps'],RawRecipe.describe()['n_ingredients'])
def wordFrequencyRecipe(visable:bool):
    textName = ' '.join(RawRecipe['name'])
    if(visable):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(textName)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Word frequency in NAME field')
        plt.axis('off')
        plt.show()

wordFrequencyRecipe(1)