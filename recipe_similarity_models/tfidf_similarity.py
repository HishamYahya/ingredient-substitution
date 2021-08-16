from .recipe_similarity import RecipeSimilarity
from vectorizers import TFIDFVectorizer

import numpy as np


class TFIDFSimilarity(RecipeSimilarity):
	def __init__(self, directory):
		super().__init__(directory)
		self.vectorizer = TFIDFVectorizer(
			model_path=f'{self.directory}/tfidf_model_ingredients_only',
			dict_path=f'{self.directory}/dictionary.txt'
		)
		self.vectors = np.loadtxt(f'{self.directory}/tfidf_vectors_ingredients_only.npy')

	