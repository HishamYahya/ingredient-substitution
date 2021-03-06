from ingredient_substitution_models import DIISHModel, IngredientSubstitution
from recipe_similarity_models import TFIDFSimilarity, RecipeSimilarity
from helper_functions import tokenize, TextCleaner
from gensim.corpora import Dictionary
from collections import defaultdict
import numpy as np
import pandas as pd
import requests


class Substitution:
    """
    Class that encapsulates everything and implements the final ``get_substitutions()``
    method.
    """
    def __init__(
        self,
        directory,
        ingredient_substitution_model: IngredientSubstitution = DIISHModel,
        recipe_similarity_model: RecipeSimilarity = TFIDFSimilarity
    ):
        """
        Parameters:
            directory (str): the path to the model files
            ingredient_substitution_model (IngredientSubstitution): the implementation of ingredient substitution
                (default is DIISHModel)
            recipe_similarity_model (RecipeSimilarity): the implementation of recipe similarity
                (default is TFIDFSimilarity)
        """
        print('Loading recipes...')
        with open(f'{directory}/recipes_ingredients_only.txt') as f:
            self.data = f.readlines()
        self.cleaner = TextCleaner(directory)
        self.generate_ghg_dict()
        self.is_model = ingredient_substitution_model(directory)
        self.rs_model = recipe_similarity_model(directory)
        self.directory = directory

    def generate_ghg_dict(self):
        """
        Loads dictionary with ghg values from the KB
        """
        print('Loading ghg dictionary...')
        self.ghg = defaultdict(float)
        ids = requests.get(
            'https://ecarekb.schlegel-online.de/foodon_ids').json()
        for ing in ids:
            name = self.cleaner.filter_ingredient(ing['ingredient'])
            if name:
                req = requests.get(
                    f'https://ecarekb.schlegel-online.de/ingredient?ingredient={"+".join(ing["ingredient"].split())}')
                ghg = req.json()['ghg']
                self.ghg[name] = ghg
            else:
                continue
            for alt_name in ing['alternate_names']:
                alt_name_f = self.cleaner.filter_ingredient(alt_name)
                if alt_name_f and alt_name_f not in self.ghg:
                    self.ghg[alt_name_f] = ghg
        print('ghg dictionary loaded!')


    

    def get_substitutions(self,
                          ingredients : [(str, bool)],
                          instructions: [str] = [],
                          k_similar_recipes:int = 100,
                          k_top_candidates: int = 5,
                          important_threshold: float = 0.8,
                          total_ghg: int = -1,
                          verbose=False):
        """
        Gets substitution suggestions using both the ingredient substitution and recipe similarity models

        Parameters:
            ingredients: a list of (ingredient, isHighCarbon) tuples
            instructions: a list of instruction strings (optional)
            k_similar_recipes: how many recipes in the cluster
            k_top_candidates: number of the top considered ingredient substitutions
            important_threshold: the minimum fraction of the similar recipes an
            ingredient occurs in for it to be considered important
            total_ghg: total carbon per kg of recipe (optional, leave blank to calculate using API ghg data)

        Returns:
        a list of dictionaries in the format of

            {
                    'from': ...
                    'to': ...
                    'confidence': ...
                    'ghg_difference': ...
                    'percent_reduction': ...
            }

        sorted by confidence
        """

        # filter ingredients and instructions and then tokenize them
        ingredients = [(self.cleaner.filter_ingredient(ing), hc) for ing, hc in ingredients]
        instructions = self.cleaner.filter_instruction(" || ".join(instructions)).split()

        # concatenate the two using @@ if there are instructions
        if instructions:
            recipe = " ".join([ing for ing, _ in ingredients]).split() + ['@@'] + instructions
        else:
            recipe = " ".join([ing for ing, _ in ingredients]).split()
        
        # get the most similar recipes
        similar_recipes = self.rs_model.get_most_similar(recipe, k=k_similar_recipes)
        # if verbose:
        #     print('Similar recipes (index, confidence):')
        #     print(similar_recipes)

        ings_only = " ".join([ing for ing, _ in ingredients]).split() 
        recipes = [self.data[index].split() for index, _ in similar_recipes
                    # only consider recipes that aren't a superset of the input recipe
                    # because we care about what can be substituted rather than added
                    if not set(self.data[index].split()).issuperset(ings_only)]

        # if verbose:
        #     print('Recipe ingredients: ', recipes)

        # get the important and substitutable ingredients
        imp, subs = self.get_substitutable_ings(
            recipes, no_above=important_threshold)

        if verbose:
            print("Important: ", imp)
            print("Substitutable: ", subs)
            print()

        substitutions = []
        # loop through every ingredient
        for ingredient, hc in ingredients:
            # if it's high carbon,
            if hc:
                for ing in ingredient.split():
                    # check if the ingredient substitution model outputs something that
                    # is substitutable in the recipe cluster
                    similar_ingredients = self.is_model.get_top_candidates(
                        ing, k=k_top_candidates)
                    for sim_ing, confidence in similar_ingredients:
                        # add it to the list of possible substitutions if it is
                        if sim_ing in subs and sim_ing not in ingredients:
                            substitutions.append(
                                {'from': ing, 'to': sim_ing, 'confidence': confidence})

        # remove duplicates
        substitutions = [dict(t)
                         for t in {tuple(s.items()) for s in substitutions}]

        # sort by how confident we are of the substitution being a viable one
        substitutions.sort(key=lambda x: x['confidence'], reverse=True)

        # calculate total ghg if not passed in
        if total_ghg == -1:
            total_ghg = self.calculate_total_ghg(ings_only)

        # only return substitutions of ingredients that are high
        # carbon and if the subtitute has a less ghg
        substitutions = list(filter(
            lambda sub: self.ghg[sub['from']] >= self.ghg[sub['to']],
            substitutions
        )
        )

        # add ghg difference and percent reduction to substitutions
        for sub in substitutions:
            sub['ghg_difference'] = self.ghg[sub['from']] - self.ghg[sub['to']]
            if total_ghg == 0:
                sub['percent_reduction'] = 0
            else:
                sub['percent_reduction'] = sub['ghg_difference'] / total_ghg * 100

        return substitutions

    def get_substitutable_ings(self, recipes, no_above=0.7):
        """
        Separates the important ingredients from the substitutable one

        Parameters:
            recipes: list of *tokenized* recipes
            no_above: the minimum fraction to be considered important

        Returns
            (important_ings, subs_ings)
        """
        id2word = Dictionary(recipes)
        all_ings = list(id2word.values())
        id2word.filter_extremes(no_below=0, no_above=no_above)
        # after filtering (substitutable)
        subs_ings = list(id2word.values())
        important_ings = list(filter(lambda x: x not in subs_ings, all_ings))
        return important_ings, subs_ings

    def calculate_total_ghg(self, ingredients: [str]):
        return sum([self.ghg[ing] for ing in ingredients])
    
    
    def get_substitutions_is_model_only(self,
                          ingredients : [(str, bool)],
                          k_top_candidates: int = 5,
                          total_ghg: int = -1):
        """
        Gets substitution suggestions using only the ingredient substitution model

        Parameters:
            ingredients: a list of (ingredient, isHighCarbon) tuples
            k_top_candidates: number of the top considered ingredient substitutions
            total_ghg: total carbon per kg of recipe (optional, leave blank to calculate using API ghg data)

        Returns:
        a list of dictionaries in the format of

            {
                    'from': ...
                    'to': ...
                    'confidence': ...
                    'ghg_difference': ...
                    'percent_reduction': ...
            }

        sorted by confidence
        """
        # calculate total ghg if not passed in
        if total_ghg == -1:
            ings_only = " ".join([ing for ing, _ in ingredients]).split() 
            total_ghg = self.calculate_total_ghg(ings_only) 
        
        subs = []
        for ing, hc in ingredients:
            name = self.cleaner.filter_ingredient(ing)
            if not name:
                continue
            if hc:
                candidates = self.is_model.get_top_candidates(name, k_top_candidates)
                for sim_ing, confidence in candidates:
                    if self.ghg[name] >= self.ghg[sim_ing]:
                        difference = self.ghg[name] - self.ghg[sim_ing] 
                        subs.append(
                                {'from': name, 'to': sim_ing, 'confidence': confidence,
                                'ghg_difference': difference,
                                'percent_reduction': (difference) / total_ghg * 100})
        
        # remove duplicates
        subs = [dict(t) for t in {tuple(s.items()) for s in subs}]

        # sort by how confident we are of the substitution being a viable one
        subs.sort(key=lambda x: x['confidence'], reverse=True)

        return subs
