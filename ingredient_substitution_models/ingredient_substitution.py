class IngredientSubstitution:
	"""
		Interface for ingredient substitution implementations
	"""
	def __init__(self, directory):
		self.directory = directory

	def get_top_candidates(self, ingredient, k):
		raise NotImplementedError('get_top_candidates')