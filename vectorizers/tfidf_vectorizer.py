import os
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
from gensim.models import TfidfModel
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class TFIDFVectorizer(BaseEstimator, TransformerMixin):

  def __init__(self, dict_path=None, model_path=None):
    self.dict_path = dict_path
    self.model_path = model_path
    self.id2word = None
    self.tfidf = None
    self.load()

  def load(self):
    if self.dict_path != None and os.path.exists(self.dict_path):
      self.id2word = Dictionary.load_from_text(self.dict_path)
    if self.model_path != None and os.path.exists(self.model_path):
      self.tfidf = TfidfModel.load(self.model_path)

  def save(self):
    if self.dict_path != None:
      self.id2word.save_as_text(self.dict_path)
    if self.model_path != None:
      self.tfidf.save(self.model_path)

  def fit(self, documents, labels=None):
    self.id2word = Dictionary(documents)
    # filter ingredients that occur less than 5 times or in more than 70% of the
    # recipes, then keep only the 1500 most frequent ingredients
    # self.id2word.filter_extremes(no_below=5, no_above=0.8, keep_n=400)
    self.tfidf = TfidfModel(dictionary=self.id2word, normalize=True)
    self.save()
    return self

  def transform(self, documents):
    for document in documents:
      docvec = self.tfidf[self.id2word.doc2bow(document)]
      yield sparse2full(docvec, len(self.id2word))