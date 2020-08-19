from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DropColumns(BaseEstimator, TransformerMixin):
  def __init__(self, columns):
    self.columns = columns
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    data = X.copy()
    return data.drop(labels=self.columns, axis='columns')

class Imputer(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    data = X.copy()
    data = pd.DataFrame(data).fillna(0)
    return data