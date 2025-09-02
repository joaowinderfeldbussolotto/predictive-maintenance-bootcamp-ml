from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd

class DropColumns(BaseEstimator, TransformerMixin):
    """
    Transformer to drop specified columns.
    """
    def __init__(self, columns_to_drop):  # Initialize with the list of columns to drop
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):  # No fitting required, return self
        return self

    def transform(self, X):  # Drop the specified columns and store remaining columns
        try:
          self.remaining_columns = X.drop(columns=self.columns_to_drop, axis=1).columns.tolist()
          return X.drop(columns=self.columns_to_drop, axis=1)
        except:
          return X

class OneHotEncoding(BaseEstimator, TransformerMixin):
    """
    Transformer for one-hot encoding categorical columns.
    """
    def __init__(self, columns):
        self.columns = columns
        self.encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')  # Avoid dummy variable trap

    def fit(self, X, y=None):
        self.encoder.fit(X[self.columns])
        self.feature_names = self.encoder.get_feature_names_out(self.columns)
        return self

    def transform(self, X):
        X = X.copy()
        encoded_data = pd.DataFrame(self.encoder.transform(X[self.columns]),
                                    columns=self.feature_names,
                                    index=X.index)  # Preserve original index
        X = X.drop(columns=self.columns)  # No index reset to prevent misalignment
        X = pd.concat([X, encoded_data], axis=1)
        return X

class ScaleFeatures(BaseEstimator, TransformerMixin):
    """
    Scales specified numerical features using MinMaxScaler.
    """
    def __init__(self, columns_to_scale):
        self.columns = columns_to_scale
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        available_cols = [col for col in self.columns if col in X.columns]  # Avoid missing column errors
        self.scaler.fit(X[available_cols])
        return self

    def transform(self, X):
        X = X.copy()
        available_cols = [col for col in self.columns if col in X.columns]  # Check columns before scaling
        X[available_cols] = self.scaler.transform(X[available_cols])
        return X
