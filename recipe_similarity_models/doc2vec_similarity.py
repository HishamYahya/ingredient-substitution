from .knn_vectors_similarity import kNNVectorsSimilarity
from vectorizers import Doc2VecVectorizer
from sklearn.neighbors import NearestNeighbors
from helper_functions import split_array_ranges, tokenize

import numpy as np


class Doc2VecSimilarity(kNNVectorsSimilarity):
	def __init__(self, directory):
		super().__init__(directory)
		self.vectorizer = Doc2VecVectorizer(f'{self.directory}/doc2vec_ingredients_and_instructions.model')
		print('Loading vectors... (this might take a while)')
		self.vectors = np.loadtxt(f'{self.directory}/doc2vec_vectors_ingredients_and_instructions.gz')
		print('Vectors loaded!')

	