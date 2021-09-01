from .knn_vectors_similarity import kNNVectorsSimilarity
from vectorizers import TFIDFVectorizer
from sklearn.neighbors import NearestNeighbors
from helper_functions import split_array_ranges, tokenize

import numpy as np


class TFIDFSimilarity(kNNVectorsSimilarity):
	def __init__(self, directory):
		super().__init__(directory)
		self.vectorizer = TFIDFVectorizer(
			model_path=f'{self.directory}/tfidf_model_ingredients_only',
			dict_path=f'{self.directory}/dictionary.txt'
		)
		print('Loading vectors... (this might take a while)')
		self.vectors = np.loadtxt(f'{self.directory}/tfidf_vectors_ingredients_only.gz')
		print('Vectors loaded!')


	