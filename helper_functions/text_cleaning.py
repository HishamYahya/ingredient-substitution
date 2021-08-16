import json
import string
import regex as re
from nltk.stem import WordNetLemmatizer

class TextCleaner:
	def __init__(self, directory):
		self.directory = directory

		with open(f'{directory}/food_names.json', 'r') as f:
			self.food_names = json.load(f)

		with open(f'{directory}/synonyms.json', 'r') as f:
			self.synonyms = json.load(f)

		self.all_names = set(self.food_names).union(self.synonyms.keys())

		self.lemmatizer = WordNetLemmatizer()

	def get_name(self, ing):
		if ing in self.food_names:
			return ing
		return self.synonyms[ing]

	def filter_ingredient(self, ing):
		ing = ing.lower()
		ing = ing.replace('-', ' ')
		ing = ing.replace(',', ' ')
		ing = ing.replace('/', ' ')

		# remove punctuation except parentheses and dashes
		ing = ing.translate(str.maketrans(
			'', '', string.punctuation.replace('()', "")))

		# remove parenthesised items
		ing = re.sub(r'\(.*?\)', "", ing)

		# remove fractions
		ing = re.sub(r'\d/\d', "", ing)

		# remove digits
		ing = re.sub(r'\d', "", ing)

		# lemmatize words
		words = [self.lemmatizer.lemmatize(word) for word in ing.split()]


		# the following loop ensures multi-word ingredient names
		# are included without including the subwords
		ing = ''
		i = 0
		while i < len(words) - 2:
			if f'{words[i]}_{words[i+1]}_{words[i+2]}' in self.all_names:
				ing += self.get_name(f'{words[i]}_{words[i+1]}_{words[i+2]}') + ' '
				i += 2
			elif f'{words[i]}_{words[i+1]}' in self.all_names:
				ing += self.get_name(f'{words[i]}_{words[i+1]}') + ' '
				i += 1
			elif f'{words[i+1]}_{words[i]}' in self.all_names:
				ing += self.get_name(f'{words[i+1]}_{words[i]}') + ' '
				i += 1
			elif words[i] in self.all_names:
				ing += self.get_name(words[i]) + " "
			i += 1
		# if there are 2 remaining words
		if i == len(words) - 2:
			if f'{words[i]}_{words[i+1]}' in self.all_names:
				ing += self.get_name(f'{words[i]}_{words[i+1]}')
			elif f'{words[i+1]}_{words[i]}' in self.all_names:
				ing += self.get_name(f'{words[i+1]}_{words[i]}')
			else:
				if words[i] in self.all_names:
					ing += self.get_name(words[i]) + ' '
				if words[i+1] in self.all_names:
					ing += self.get_name(words[i+1])

		# if there's 1 remaining word
		if i == len(words) - 1:
			if words[i] in self.all_names:
				ing += self.get_name(words[i])

		return " ".join(ing.split())
