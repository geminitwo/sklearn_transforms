from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin

	# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
	def __init__(self, columns):
	    self.columns = columns

	def fit(self, X, y=None):
	    return self

	def transform(self, X):
	    # Primeiro realizamos a cópia do dataframe 'X' de entrada
	   	data = X.copy()
	    # Retornamos um novo dataframe sem as colunas indesejadas
	    return data.drop(labels=self.columns, axis='columns')

class Smote(BaseEstimator, TransformerMixin):
	def __init__(self, columns):
	    self.columns = columns

	def fit(self, X, y):
		X, y = SMOTE(random_state=0).fit_resample(X, y)
		return X, y

	def transform(self, X):
	    pass