from ingredient_substitution_models import DIISHModel, IngredientSubstitution
from recipe_similarity_models import TFIDFSimilarity, RecipeSimilarity
from helper_functions import tokenize, TextCleaner
from gensim.corpora import Dictionary

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


	"""
	Parameters:
	recipe: a list of ingredient strings, formatting doesn't matter

	Returns:
	a list of tuples in the format of
	(original_ingredient, substitution, confidence)
	sorted by confidence
	"""
	def get_substitutions(self, recipe, verbose=False):
		# filter every ingredient and join them into one string
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

directory = '/Users/hisham/Google Drive/Recipes1M'
sub = Substitution(directory)

recipe = [
	'6 ounces penne',
	'2 cups Beechers Flagship Cheese Sauce (recipe follows)',
	'1 ounce Cheddar, grated (1/4 cup)',
	'1 ounce Gruyere cheese, grated (1/4 cup)',
	'1/4 to 1/2 teaspoon chipotle chili powder (see Note)',
	'1/4 cup (1/2 stick) unsalted butter',
	'1/3 cup all-purpose flour',
	'3 cups milk',
	'14 ounces semihard cheese (page 23), grated (about 3 1/2 cups)',
	'2 ounces semisoft cheese (page 23), grated (1/2 cup)',
	'1/2 teaspoon kosher salt',
	'1/4 to 1/2 teaspoon chipotle chili powder',
	'1/8 teaspoon garlic powder',
	'(makes about 4 cups)'
 ]

print(sub.get_substitutions(recipe, verbose=True))