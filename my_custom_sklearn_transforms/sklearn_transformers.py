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
class Smote(BaseEstimator, TransformerMixin):
	def __init__(self, columns):
		self.columns = columns
	def fit(self, X, y):
		X, y = SMOTE(random_state=0).fit_resample(X, y)
		return X, y
	def transform(self, X):
		pass