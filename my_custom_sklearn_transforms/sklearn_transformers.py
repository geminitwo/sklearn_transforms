from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin

class DropColumns(BaseEstimator, TransformerMixin):
	def __init__(self, columns):
		self.columns = columns
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		data = X.copy()
		return data.drop(labels=self.columns, axis='columns')

class SmoteBel(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None):
		X, y = SMOTE(random_state=2).fit_sample(X, y)
		return self
	def transform(self, X):
		data = X.copy()
		return data