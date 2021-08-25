# pylint: disable=import-error
from .ingredient_substitution import IngredientSubstitution
import numpy as np
from gensim.corpora import Dictionary

class DIISHModel(IngredientSubstitution):
	def __init__(self, directory):
		super().__init__(directory)
		self.matrix = np.loadtxt(f'{self.directory}/DIISH_matrix.npy')
		self.dictionary = Dictionary.load_from_text(f'{self.directory}/dictionary.txt')

	def get_top_candidates(self, ingredient, k=10):
		scores = self.matrix[self.dictionary.token2id[ingredient]]
		scores = [(self.dictionary[i], score/4.5) for i, score in enumerate(scores) if score == score]
		return sorted(scores, key=lambda x: x[1], reverse=True)[1:k+1]
