import sys
import os
import requests
import json
import warnings
import spacy
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.test.utils import datapath
from scipy.spatial import distance
from itertools import combinations
from helper_functions import TextCleaner, tokenize
from vectorizers import TFIDFVectorizer

def main(argv):
	if not os.path.exists('build'):
		os.mkdir('build')
	if not argv:
		print('Please provide the path of Recipe1M layer1.json')
		exit()
	path = argv[0]
	
	print('---- Generating all needed files ---- this might take upwards of 4 hours')
	generate_food_names_and_synonyms()
	generate_filtered_recipe_dataset(path)
	generate_dictionary()
	generate_cooccurrence_matrix()
	generate_word2vec_model()
	generate_fc_matrix()
	generate_fic_vectors()
	generate_DIISH_matrix()
	generate_tfidf_recipe_similarity_model_and_vectors()
	

def generate_food_names_and_synonyms():
	print('Generating food names...')

	ids = requests.get('https://ecarekb.schlegel-online.de/foodon_ids').json()
	cleaner = TextCleaner()
	food_names = set([cleaner.normalise_ingredient(ing['ingredient']) for ing in ids])
	with open('build/food_names.json', 'w') as f:
		json.dump(list(food_names), f)

	synonyms = dict()
	for ing in ids:
		name = cleaner.normalise_ingredient(ing['ingredient'])
		syns = [
			cleaner.normalise_ingredient(word)
			for word in ing['alternate_names']
			if cleaner.normalise_ingredient(word) != name
		]
		for word in syns:
			synonyms[word] = name
	with open('build/synonyms.json', 'w') as f:
			json.dump(synonyms, f)


def generate_filtered_recipe_dataset(layer_path):
	"""
	Pass in the path of Recipe1M's layer1.json
	"""
	import json
	try:
		cleaner = TextCleaner('build')
	except:
		raise FileNotFoundError('Please ensure food_names.json and synonyms.json have been generated and are in the "build" directory')
		
	print('Loading Recipe1M...')
	with open(layer_path, 'r') as f:
		data = json.load(f)
	print('Data loaded!')

	def to_recipe_string_list(recipes, with_instructions=False):
		"""
		Generator that turns every recipe into a string of its ingredients
		"""
		for recipe in recipes:
			recipe_ings = []
			for ing in recipe['ingredients']:
				filtered_ing = cleaner.filter_ingredient(ing['text'])
				if filtered_ing:
					recipe_ings.append(filtered_ing)
			ing_string = " ".join(recipe_ings)
			if with_instructions:
				recipe_insts = []
				for inst in recipe['instructions']:
					recipe_insts.append(cleaner.filter_instruction(inst['text']))
					instructions_string = " || ".join(recipe_insts)
				yield ing_string + " @@ " + instructions_string
			else:
				yield ing_string
	
	with open('build/recipes_ingredients_and_instructions.txt', 'a') as f1:
		with open('build/recipes_ingredients_only.txt', 'a') as f2:
			f1.seek(0)
			f1.truncate()
			f2.seek(0)
			f2.truncate()
			i = 0
			for rec in to_recipe_string_list(data, with_instructions=True):
				print(f'Generating filtered recipe datasets... {int(i/1029720 * 100)}% done', end='\r')
				ingredients = rec.split('@@')[0].strip()
				f1.write(rec + '\n')
				f2.write(ingredients + '\n')
				i += 1
			print(f'Generating recipe datasets... 100% done')
	print('Datasets generated!')


def generate_dictionary():
	try:
		with open('build/recipes_ingredients_only.txt', 'r') as f:
			recipes = f.readlines()
			recipes = [line.split() for line in recipes]
	except FileNotFoundError:
		raise FileNotFoundError('Make sure to generate the dataset first using generate_filtered_recipe_dataset().')
	
	Dictionary(documents=recipes).save_as_text('build/dictionary.txt')

def load_dictionary() -> Dictionary:
	try:
		return Dictionary.load_from_text('build/dictionary.txt')
	except FileNotFoundError:
		raise FileNotFoundError('Make sure to generate the dictionary first.')


def generate_word2vec_model():
	try:
		sentences = LineSentence(datapath(os.path.abspath('build/recipes_ingredients_and_instructions.txt')))
	except FileNotFoundError:
		raise FileNotFoundError('Make sure to generate the dataset first using generate_filtered_recipe_dataset().')
	print('Training word2vec model...')
	model =	Word2Vec(sentences=sentences)
	print('Saving model...')
	model.save('build/word2vec.model')
	print('Model saved!')


def generate_cooccurrence_matrix():
	'''
	Saves a file for each word containing the co-occurrence vector.
	(used to speed up D execution)
	'''
	dictionary = load_dictionary()

	try:
		with open('build/recipes_ingredients_only.txt', 'r') as f:
			matrix = []
			for i in range(len(dictionary)):
				print(f'Generating co-occurrence matrix... {i}/{len(dictionary)} words done', end='\r')
				word = dictionary[i]
				vector = np.zeros(len(dictionary))
				recipe_count = 0

				for i, line in enumerate(f):
					ings = line.split()
					if word in ings:
						recipe_count += 1
						for ing in ings:
							vector[dictionary.token2id[ing]] += 1

				if recipe_count != 0:
					vector = vector/recipe_count

				matrix.append(vector)
				f.seek(0)
			print(f'Generating co-occurrence matrix... {len(dictionary)}/{len(dictionary)} words done')
	except FileNotFoundError:
		raise FileNotFoundError('Make sure to generate the dataset first.')
	
	print('Saving...')
	np.savetxt('build/cooccurrence_matrix.npy', matrix)
	print('Co-occurrence matrix saved!')


def generate_fc_matrix():
	"""
	Context counts matrix
	"""
	dictionary = load_dictionary()
	fc = np.zeros((len(dictionary), len(dictionary)))
	try:
		with open('build/recipes_ingredients_only.txt', 'r') as f:
			for i, line in enumerate(f):
				print(f'Generating fc matrix... {int(i/1029720 * 100)}% done', end='\r')
				ings = line.split()
				for a, b in combinations(ings, 2):
					fc[dictionary.token2id[a]][dictionary.token2id[b]] += 1
		print(f'Generating fc matrix... 100% done')
	except FileNotFoundError:
		raise FileNotFoundError('Make sure to generate the dataset first.')
	
	print('Saving...')
	np.savetxt('build/fc_matrix.npy', fc)
	print('Fc matrix saved!')


def generate_fic_vectors():
	if not os.path.exists('build/fic_vectors'):
		os.mkdir('build/fic_vectors')

	dictionary = load_dictionary()

	with open('build/recipes_ingredients_only.txt', 'r') as f:
		for i in range(len(dictionary)):
			print(f'Generating fic vectors... {i}/{len(dictionary)} words done', end='\r')
			word = dictionary[i]
			fic = np.zeros((len(dictionary), len(dictionary)))
			for _, line in enumerate(f):
				ings = line.split()
				if word in ings:
					for context in combinations(ings, 2):
						index = (dictionary.token2id[context[0]],dictionary.token2id[context[1]])
						fic[index] += 1
			np.savetxt(f'build/fic_vectors/{word}.npy', fic)
			f.seek(0)
	print(f'Generating fic vectors... {len(dictionary)}/{len(dictionary)} words done')
	print('Fic vectors saved!')

def get_fic_matrix():
	dictionary = load_dictionary()

	matrix = []
	for i in range(len(dictionary)):
		try:
			m = np.loadtxt(f'build/fic_vectors/{dictionary[i]}.npy')
		except:
			raise FileNotFoundError(f'Vectors incomplete, {dictionary[i]}.npy not found')
		matrix.append(m)
	return np.array(matrix)


def generate_DIISH_matrix():
	print('Initialising DIISH...')
	diish = DIISH()
	if diish.dictionary is None:
		raise FileNotFoundError('dictionary.txt failed to load.')
	print('DIISH initialised.')

	DIISH_matrix = np.zeros((len(diish.dictionary), len(diish.dictionary)))
	for i in range(len(diish.dictionary)):
		print(f'Generating DIISH matrix... {i}/{len(diish.dictionary)} words done', end='\r')
		for j in range(len(diish.dictionary)):
			DIISH_matrix[i][j] = diish(diish.dictionary[i], diish.dictionary[j])
	print(f'Generating DIISH matrix... {len(diish.dictionary)}/{len(diish.dictionary)} words done', end='\r')
	print('Matrix generated!')
	print('Saving...')
	np.savetxt('build/DIISH_matrix.npy', DIISH_matrix)
	print('DIISH matrix saved!')

class DIISH:
	def __init__(self, directory='build'):
		self.directory = directory
		try:
			self.dictionary = Dictionary.load_from_text(f'{directory}/dictionary.txt')
			nlp = spacy.load('en_core_web_lg')
			self.nlps = dict()
			for ing in self.dictionary.token2id:
				ing = ing.replace('_', ' ')
				self.nlps[ing] = nlp(ing)
		except FileNotFoundError:
			warnings.warn('dictionary.txt not in directory, can\'t calculate S and D')
			self.dictionary = None
		try:
			self.word2vec = Word2Vec.load(f'{directory}/word2vec.model')
		except FileNotFoundError:
			warnings.warn('word2vec.model not in directory, can\'t calculate W')
			self.word2vec = None
		try:
			self.co_occ = np.loadtxt(f'{directory}/cooccurrence_matrix.npy')
		except FileNotFoundError:
			warnings.warn('cooccurrence_matrix.npy not in directory, can calculate D but will be a lot slower (provided recipes_ingredients_only.txt is in the directory)')
			self.co_occ = None
		try:
			self.fc = np.loadtxt(f'{directory}/fc_matrix.npy')
		except FileNotFoundError:
			warnings.warn('fc_matrix.npy not in directory, can\'t calculate P')
			self.fc = None
		try:
			self.fic = get_fic_matrix()
		except FileNotFoundError as e:
			print(e)
			warnings.warn('Failed to load the fic matrix, can\'t calculate P')
			self.fic = None
		

	def W(self, a, b):
		if a not in self.word2vec.wv.key_to_index or b not in self.word2vec.wv.key_to_index:
			return 0
		return self.word2vec.wv.similarity(a, b)

	def S(self, a, b):
		if self.nlps is None:
			print('Dictionary not loaded, can\'t calculate S')
			return None
		# for multi-word ingredients
		a = a.replace('_', ' ')
		b = b.replace('_', ' ')
		return self.nlps[a].similarity(self.nlps[b])

	def D(self, a, b):
		if not self.co_occ is None:
			a_vector = self.co_occ[self.dictionary.token2id[a]]
			b_vector = self.co_occ[self.dictionary.token2id[b]]	
		else:
			a_vector, b_vector = np.zeros(len(self.dictionary)), np.zeros(len(self.dictionary))
			a_recipe_count, b_recipe_count = 0, 0
			with open(f'{self.directory}/recipes_ingredients_only.txt', 'r') as f:
				for _, line in enumerate(f):
					ings = line.split()
					if a in ings:
						a_recipe_count += 1
						for ing in ings:
							a_vector[self.dictionary.token2id[ing]] += 1

					if b in ings:
						b_recipe_count += 1
						for ing in ings:
							b_vector[self.dictionary.token2id[ing]] += 1

			if a_recipe_count != 0:
				a_vector = a_vector/a_recipe_count

			if b_recipe_count != 0:
				b_vector = b_vector/b_recipe_count

		if np.count_nonzero(a_vector) == 0 or np.count_nonzero(b_vector) == 0:
			return 0

		return 1 - distance.cosine(a_vector, b_vector)	
	
	def PPMI(self, fic, fi, fc):
		b = fi * fc
		return np.maximum(np.log10(np.divide(fic * len(self.dictionary) * len(fc), b, out=np.zeros(fic.shape, dtype=float), where=b!=0))
		* np.sqrt(np.maximum(fi, fc)), np.zeros(len(fc)))

	def P(self, a, b):
		if self.fic is None:
			print('Fic matrix not loaded, can\'t calculate P')
			return None 
		if self.fc is None:
			print('Fc matrix not loaded, can\'t calculate P')
			return None 
		
		fic_a = self.fic[self.dictionary.token2id[a]]
		fic_b = self.fic[self.dictionary.token2id[b]]

		fi_a = self.dictionary.dfs[self.dictionary.token2id[a]]
		fi_b = self.dictionary.dfs[self.dictionary.token2id[b]]

		ppmi = self.PPMI(fic_a.flatten(), fi_a, self.fc.flatten()), self.PPMI(fic_b.flatten(), fi_b, self.fc.flatten())

		if np.count_nonzero(ppmi[0]) == 0 or np.count_nonzero(ppmi[1]) == 0:
			return 0
		
		return 1 - distance.cosine(ppmi[0], ppmi[1])
	
	def __call__(self, a, b):
		if self.fc is None or self.fic is None or self.dictionary is None or self.word2vec is None:
			print('Can\'t calculate DIISH, make sure all needed files are present in the directory')	
			return None
		return self.W(a, b) + (self.S(a, b) ** 2) + (0.5 * self.D(a, b) ** 0.25) + (2 * self.P(a, b) ** 0.5)


def generate_tfidf_recipe_similarity_model_and_vectors():
	print('--- TFIDF Model and Vectors ---')
	try:
		print('Loading recipes...')
		with open('build/recipes_ingredients_only.txt', 'r') as f:
			data = f.readlines()
	except FileNotFoundError:
		raise FileNotFoundError('Make sure the datasets have been generated')
	
	print('Tokenizing data...')
	data = list(tokenize(data))
	print('Fitting model...')
	model = TFIDFVectorizer().fit(data)
	print('Saving model...')
	model.tfidf.save('build/tfidf_model_ingredients_only')

	print('Generating vectors...')
	vecs = list(model.transform(data))
	print('Saving vectors...')
	np.savetxt('build/tfidf_vectors_ingredients_only.gz', vecs)
	print('Done!')


if __name__ == '__main__':
	main(sys.argv)

