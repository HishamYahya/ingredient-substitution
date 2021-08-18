from ingredient_substitution_models import DIISHModel, IngredientSubstitution
from recipe_similarity_models import TFIDFSimilarity, RecipeSimilarity
from helper_functions import tokenize, TextCleaner
from gensim.corpora import Dictionary
from collections import defaultdict
import numpy as np
import pandas as pd


class Substitution:
	def __init__(
			self,
			directory,
			ingredient_substitution_model: IngredientSubstitution = DIISHModel,
			recipe_similarity_model: RecipeSimilarity = TFIDFSimilarity
	):
		self.is_model = ingredient_substitution_model(directory)
		self.rs_model = recipe_similarity_model(directory)
		self.cleaner = TextCleaner(directory)
		self.directory = directory

		self.generate_ghg_dict()


	def generate_ghg_dict(self):
		df = pd.read_excel(f'{self.directory}/Master data - ECare - v12.xlsx', sheet_name='Master DB')
		ghg = np.array(df.iloc[:,[0, 1, 8]][df['Food'].notna()])
		ghg = [(self.cleaner.normalise_ingredient(x), self.cleaner.normalise_ingredient(y), z) for x, y, z in ghg]
		self.ghg = defaultdict(int)
		for ing, syn, val in ghg:
			self.ghg[ing] = val
		if syn == syn:
			self.ghg[syn] = val

	"""
	Parameters:
	recipe: a list of ingredient strings, formatting doesn't matter

	Returns:
	a list of tuples in the format of
	(original_ingredient, substitution, confidence)
	sorted by confidence
	"""
	def get_substitutions(self, recipe, verbose=False):
		# filter every ingredient and join them into one string then split
		# in case more than one ingredient is in a list element
		recipe = " ".join([self.cleaner.filter_ingredient(ing) for ing in recipe])

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
			print(recipes)
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

		# add ghg difference to substitutions
		for sub in substitutions:
			sub['ghg_difference'] = self.ghg[sub['from']] - self.ghg[sub['to']]

		# change to dictionary


		return substitutions

	"""
	Seperates the important ingredients from the substituable one

	Parameters:
	recipes: list of *tokenized* recipes
	no_above: the minimum fraction to be considered important

	Returns
	important_ings
	subs_ings
	"""
	def get_substitutable_ings(self, recipes, no_above = 0.6):
		id2word = Dictionary(recipes)
		all_ings = list(id2word.values())
		id2word.filter_extremes(no_below=0, no_above=no_above)
		# after filtering (substitutable)
		subs_ings = list(id2word.values())
		important_ings = list(filter(lambda x: x not in subs_ings, all_ings))
		return important_ings, subs_ings
