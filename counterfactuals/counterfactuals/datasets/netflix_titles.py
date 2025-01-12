import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from counterfactuals.datasets.base import AbstractDataset


class NetflixDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/netflix_titles.csv"):
        self.raw_data = self.load(file_path=file_path)
        self.X, self.y = self.preprocess(raw_data=self.raw_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
            self.X, self.y
        )
        self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
            self.X_train, self.X_test, self.y_train, self.y_test
        )

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        self.feature_columns = ["release_year", "duration", "rating"]
        target_column = "type"  
        raw_data = raw_data.dropna(subset=self.feature_columns)

        raw_data[target_column] = raw_data[target_column].map({"Movie": 0, "TV Show": 1})

        def parse_duration(value):
            if "min" in value:
                return int(value.replace(" min", ""))
            elif "Season" in value:
                return int(value.split(" ")[0]) * 60  
            return 0

        raw_data["duration"] = raw_data["duration"].apply(parse_duration)

        self.numerical_columns = [0, 1] 
        self.categorical_columns = [2] 

        X = raw_data[self.feature_columns].to_numpy()
        y = raw_data[target_column].to_numpy()
        return X, y

    def transform(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ):

        self.feature_transformer = ColumnTransformer(
            [
                ("MinMaxScaler", MinMaxScaler(), self.numerical_columns),
                (
                    "OneHotEncoder",
                    OneHotEncoder(sparse_output=False),
                    self.categorical_columns,
                ),
            ],
        )
        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        self.numerical_features = list(range(0, len(self.numerical_columns)))
        self.categorical_features = list(
            range(len(self.numerical_columns), X_train.shape[1])
        )
        self.actionable_features = list(range(0, X_train.shape[1]))

        return X_train, X_test, y_train, y_test
