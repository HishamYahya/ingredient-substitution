from ingredient_substitution_models import DIISHModel
from recipe_similarity_models import TFIDFSimilarity
from helper_functions import tokenize
from gensim.corpora import Dictionary

class Substitution:
	def __init__(
			self,
			directory,
			ingredient_substitution_model=DIISHModel,
			recipe_similarity_model=TFIDFSimilarity
	):
		self.is_model = ingredient_substitution_model(directory)
		self.rs_model = recipe_similarity_model(directory)
		self.directory = directory


	"""
	Parameters:
	recipe: a list of ingredient strings, formatting doesn't matter

	Returns:
	a list of tuples in the format of
	(original_ingredient, substitution, confidence)
	sorted by confidence
	"""
	def get_substitutions(self, recipe, verbose=False):
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
		# loop through every ingredients in the passed recipe
		for ingredient in recipe:
			# if it's substitutable in the recipe,
			if ingredient in subs:
				# check if the FastText model outputs something that
				# is also substitutable in the recipe
				similar_ingredients = self.is_model.get_top_candidates(ingredient, k=5)
				for sim_ing, confidence in similar_ingredients:
					# add it to the list of possible substitutions if it is
					if sim_ing in subs and sim_ing not in recipe:
						substitutions.append((ingredient, sim_ing, confidence))

		# remove duplicates
		substitutions = list(set(substitutions))
		# sort by how confident we are of the substitution being a viable one
		substitutions.sort(key=lambda x: x[2], reverse=True)

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

sub = Substitution('/Users/hisham/Google Drive/Recipes1M')
print(sub.get_substitutions('cheese cheddar gruyere cheese chilli butter flour milk cheese cheese salt chilli garlic', verbose=True))