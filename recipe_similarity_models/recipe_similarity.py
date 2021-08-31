"""
	Interface for recipe similarity implementations
"""
class RecipeSimilarity:
	def __init__(self, directory):
		self.directory = directory

	def get_most_similar(self, recipe: [str], k = 10, n_clusters = 10):
		raise NotImplementedError('get_most_similar is not implemented')