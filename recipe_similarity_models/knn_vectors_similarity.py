from .recipe_similarity import RecipeSimilarity
from vectorizers import Doc2VecVectorizer
from helper_functions import split_array_ranges, tokenize
from sklearn.neighbors import NearestNeighbors

import numpy as np


class kNNVectorsSimilarity(RecipeSimilarity):
	"""
	Interface for vector-based, kNN recipe similarity implementations
	"""
	def __init__(self, directory):
		super().__init__(directory)
		self.vectorizer = None
		self.vectors = []

	def get_most_similar(self, recipe: [str], k = 10, n_clusters = 10):
		# get vector of recipe
		docvec = next(self.vectorizer.transform([recipe]))

		# cut data to n_clusters number of clusters
		similar_recipes = []
		for start, end in split_array_ranges(len(self.vectors), n_clusters):
			if end - start < k:
				break
			nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='auto').fit(self.vectors[start:end])
			distances, indicies = nbrs.kneighbors([docvec])
			indicies = list(map(lambda x: x+start, indicies))
			for x in zip(indicies[0], distances[0]):
				similar_recipes.append(x)
		return sorted(similar_recipes, key=lambda x: x[1])[:k]

	