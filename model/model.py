import pickle
import timy
import pandas as pd
import numpy as np
from dataset.trackid_to_data import create_mapping
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


class Model:

    def __init__(self, dataset_path, model_path=None) -> None:
        self.load_dataset(dataset_path=dataset_path)
        self.load_model(model_path=model_path)
        self.song_mapping = create_mapping()

    @timy.timer(ident="Loading Dataset")
    def load_dataset(self, dataset_path):
        dataset = pd.read_csv(dataset_path, dtype=int)
        if dataset.isnull().any(axis=None):
            raise ValueError("Sparse Dataset")
        self.__dataset = dataset

    def load_model(self, model_path=None):
        if model_path:
            with timy.Timer("Loading Model"):
                with open(model_path, "rb") as model_file:
                    self.__model = pickle.load(model_file)
        else:
            with timy.Timer("Creating new model"):
                self.__model = GradientBoostingClassifier(
                    learning_rate=0.01,
                    max_depth=3,
                    max_features=1.0,
                    min_samples_leaf=20,
                    min_samples_split=8,
                    n_estimators=100,
                    subsample=0.8,
                )

    @timy.timer(ident="Saving Model")
    def save_model(self, model_path="model.pkl"):
        with open(model_path, "wb") as model_file:
            pickle.dump(self.__model, model_file)

    def train(self):
        with timy.Timer("Preparing dataset for training"):
            X = self.__dataset.drop(["ItemID"], axis=1).values
            Y = self.__dataset.ItemID.values

            X_train, _, y_train, _ = train_test_split(
                X, Y, test_size=0.20, random_state=1, stratify=Y
            )

        with timy.Timer("Training"):
            self.__model.fit(X_train, y_train)

    def predict(self, args, no_preds=5):
        with timy.Timer("Performing inference"):
            preds = self.__model.predict_proba(args)
        with timy.Timer("Transforming results"):
            classes_preds = [dict(zip(self.__model.classes_, pred)) for pred in preds]
            sorted_preds = np.array([
                sorted(cls_pred, key=cls_pred.get, reverse=True)
                for cls_pred in classes_preds
            ])
        truncated_preds = sorted_preds[:,:no_preds].tolist()
        return [list(map(self.song_mapping, pred)) for pred in truncated_preds]
