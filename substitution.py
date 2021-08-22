from ingredient_substitution_models import DIISHModel, IngredientSubstitution
from recipe_similarity_models import TFIDFSimilarity, RecipeSimilarity
from helper_functions import tokenize, TextCleaner
from gensim.corpora import Dictionary
from collections import defaultdict
import numpy as np
import pandas as pd
import requests


class Substitution:
	def __init__(
			self,
			directory,
			ingredient_substitution_model: IngredientSubstitution = DIISHModel,
			recipe_similarity_model: RecipeSimilarity = TFIDFSimilarity
	):
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
		ids = requests.get('https://ecarekb.schlegel-online.de/foodon_ids').json()
		for ing in ids:
			name = self.cleaner.filter_ingredient(ing['ingredient'])
			if name:
				req = requests.get(f'https://ecarekb.schlegel-online.de/ingredient?ingredient={"+".join(ing["ingredient"].split())}')
				ghg = req.json()['ghg']
				self.ghg[name] = ghg
			else:
				continue
			for alt_name in ing['alternate_names']:
				alt_name_f = self.cleaner.filter_ingredient(alt_name)
				if alt_name_f and alt_name_f not in self.ghg:
					self.ghg[alt_name_f] = ghg
		print('ghg dictionary loaded!')


	def get_substitutions(self, recipe, verbose=False):
		"""
		Parameters:
		recipe: a list of ingredient strings, formatting doesn't matter

		Returns:
		a list of dictionaries in the format of

		{
			'from': ...
			'to': ...
			'confidence': ...
			'ghg_difference': ...
			'percent_decrease': ...
		}
		
		sorted by confidence
		"""
		# filter every ingredient and join them into one string then split
		# in case more than one ingredient is in a list element
		recipe = self.cleaner.filter_ingredient(" ".join(recipe))

		# get the most similar recipes
		similar_recipes = self.rs_model.get_most_similar(recipe)
		if verbose:
			print('Similar recipes (index, confidence):')
			print(similar_recipes)

		# load the recipes' ingredients and tokenize them
		recipes_ind = [x[0] for x in similar_recipes]
		recipes = []
		with open(f'{self.directory}/recipes_ingredients_only.txt') as f:
		  for i, line in enumerate(f):
		    if i in recipes_ind:
		      recipes.append(line.split())

		# recipes = [data[index].split() for index, _ in similar_recipes]

		if verbose:
			print('Recipe ingredients: ', recipes)
		# get the important and substitutable ingredients
		imp, subs = self.get_substitutable_ings(recipes)

		if verbose:
			print("Important: ", imp)  
			print("Substitutable: ", subs)  

		substitutions = []
		# loop through every ingredient in the passed recipe
		for ingredient in recipe.split():
			# if it's substitutable in the recipe,
			if ingredient in subs:
				# check if the ingredient substitution model outputs something that
				# is also substitutable in the recipe
				similar_ingredients = self.is_model.get_top_candidates(ingredient, k=5)
				for sim_ing, confidence in similar_ingredients:
					# add it to the list of possible substitutions if it is
					if sim_ing in subs and sim_ing not in recipe:
						substitutions.append({'from': ingredient, 'to': sim_ing, 'confidence': confidence})

		# remove duplicates
		substitutions = [dict(t) for t in {tuple(s.items()) for s in substitutions}]
		# sort by how confident we are of the substitution being a viable one
		substitutions.sort(key=lambda x: x['confidence'], reverse=True)

		total_ghg = sum([self.ghg[ing] for ing in recipe])

		# only return substitutions of ingredients that make up >=20%
		# of the total recipe's ghg and if the subtitute has a less ghg
		substitutions = list(filter(
			lambda sub: self.ghg[sub['from']] > self.ghg[sub['to']] and
			self.ghg[sub['from']] >= 0.2 * total_ghg,
			substitutions
			)
		)

		# add ghg difference and percent decrease to substitutions
		for sub in substitutions:
			sub['ghg_difference'] = self.ghg[sub['from']] - self.ghg[sub['to']]
			sub['percent_decrease'] = sub['ghg_difference'] / total_ghg

		return substitutions

	def get_substitutable_ings(self, recipes, no_above = 0.6):
		"""
		Seperates the important ingredients from the substituable one

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
