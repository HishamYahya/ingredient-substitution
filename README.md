# Unsupervised Ingredient Substitution Model
Ingredient substitution suggestions that minimize the carbon footprint of a given recipe.
# Description
The model is trained on [Recipe1M](http://pic2recipe.csail.mit.edu/), a dataset containing 1 million+ recipes with their ingredients and instructions included (amongst other things we wouldn't use). A list of the defined ingredients (the ones that have their carbon emissions data available) is retrieved from the KB (https://ecarekb.schlegel-online.de/foodon_ids) and an ingredient name dictionary is generated alongside a synonym translation map, both are then used to normalize the ingredient names in the generation of the training corpora and in the preprocessing step during the prediction.

## Training corpora
Two training corpora are generated:
* **recipes_ingredients_only.txt**

    Each recipe is represented by a concatenation of its ingredients. A filtering process is underwent that ensures that only defined ingredients are present in the corpus.

    This dataset is used to:
    * train the TF-IDF version of the recipe similarity model.
    * calculate the D and P scores.
* **recipes_ingredients_and_instructions.txt**
  
    Same as the recipes_ingredients_only.txt dataset, only the instructions are appended to the end of each recipe. A normalization process is gone through that ensures all recognized ingredient names in the instructions are normalized.

    This dataset is used to:
    * train the Doc2Vec version of the recipe similarity model.
    * train the Word2Vec model used in calculating the W score.

## Ingredient substitution ranking (DIISH)
The bulk of the suggestion work is done using a [DIISH based ranking system](https://www.frontiersin.org/articles/10.3389/frai.2020.621766/full). The implementation of the work done on this paper is presented in the notebook ```ingredient_substitution.ipynb```. A matrix is generated that contains the DIISH scores between each pair of the defined ingredients. Prediction is done by retrieving the row of the input ingredient and sorting them in descending order.

## Recipe similarity
Recipe similarity is used to find out which ingredients occur in other similar recipes and, from there, filter out the substitutions that aren't used in any of them.

Recipe similarity is implemented through vectorization and cosine distance. The following are the 2 implemented approaches:
* **TF-IDF**:
  Representing each recipe as a vector of size n (the number of defined ingredients) with their corresponding TF-IDF values. *(Doesn't take advantage of instructions)*

* **Doc2Vec**
  [Gensim's Doc2Vec model](https://radimrehurek.com/gensim/models/doc2vec.html) trained on the **recipes_ingredients_and_instructions.txt** corpus

## Prediction process run-through
The mechanism for the prediction is as follows: 

the user inputs a recipe (a list of ```(ingredient, is_high_carbon)``` tuples and, optionally, instructions) that is then normalized and vectorized using either a TF-IDF or Doc2Vec model trained on Recipe1M. The pre-computed vectors of all of Recipe1M's recipes are then queried to find the k most similar recipes. 

The ingredients of this cluster of recipes are then retrieved and split into 2 sets: important and substitutable ingredients based on how many of the recipes each ingredient occurred in (default is 80%). 

Then, for every high carbon ingredient in the input recipe, using the ingredient substitution model (currently only DIISH is implemented), the n most similar *ingredients* are retrieved. Going through the n ingredients, only the ones that occur in the *substitutable* list are considered valid substitutions and are then added to the output substitution list. 

The final step is to filter out the substitutions that have a higher carbon/kg value than the original ingredient (i.e. if the substitution increases the total carbon footprint of the recipe).




# Getting Started

## Generating needed files and training
You'll first have to have Recipe1M's layer1.json downloaded.
Running 
```
python3 generate_model.py path/to/layer1.json
```
then will generate all needed files in a folder called "build" in the project directory. Alternatively, if you want to generate only specific parts of the model, the process is split into functions that can be called independently. Some steps require previous files to have been generated though so make sure all needed files for the files you're looking to generate are in the "build" folder.


## Getting substitution suggestions
Create an instance of the Substitution class, passing it the path of the directory containing the needed files. It takes around 15 minutes for it to initialize. You can then pass a list of ```(ingredient, is_high_carbon)``` tuples and, optionally, instructions to ```get_substitutions()```
to get suggestions in the form of:
```
{
  'from': ...
  'to': ...
  'confidence': ...
  'ghg_difference': ...
  'percent_reduction': ...
}
```

## Demo
Take a look at demo.py for a simple example application of the model. It takes in a string of ingredients (for example “flour cinnamon salt baking powder egg sugar vegetable oil vanilla walnut”) of any format and outputs the suggestions.
