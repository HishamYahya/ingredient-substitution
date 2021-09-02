import os
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time


class Doc2VecVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, path=None):
        self.path = path
        self.model = None
        self.load()

    def load(self):
        if self.path != None:
            self.model = Doc2Vec.load(self.path)

    def save(self, path):
        self.model.save(path)

    def fit(self,
            documents=None,
            corpus_file=None,
            vector_size=600,
            min_count=5,
            seed=1,
            workers=2):
        if corpus_file is None:
            corpus = [
                TaggedDocument(words, [idx])
                for idx, words in enumerate(documents)
            ]
            self.model = Doc2Vec(corpus,
                                 vector_size=vector_size,
                                 min_count=min_count,
                                 seed=seed,
                                 workers=workers)
        else:
            self.model = Doc2Vec(corpus_file=corpus_file,
                                 vector_size=vector_size,
                                 min_count=min_count,
                                 seed=seed,
                                 workers=workers)
        return self

    def transform(self, documents):
        for document in documents:
            yield self.model.infer_vector(document)
