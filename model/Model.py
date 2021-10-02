import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self, dataset, model_path=None) -> None:
        self.__dataset = dataset
        self.__model = self.load_model(model_path=model_path)

    def load_model(self, model_path=None):
        if model_path:
            with open(model_path, "rb") as model_file:
                self.__model = pickle.load(model_file)
        else:
            self.__model = GradientBoostingClassifier(
                learning_rate=0.01,
                max_depth=3,
                max_features=1.0,
                min_samples_leaf=20,
                min_samples_split=8,
                n_estimators=100,
                subsample=0.8,
            )

    def save_model(self, model_path):
        with open(model_path, "wb") as model_file:
            pickle.dump(self.__model, model_file)

    def train(self):
        X = self.__dataset.drop(["ItemID"], axis=1).values
        Y = self.__dataset.ItemID.values

        X_train, _, y_train, _ = train_test_split(
            X, Y, test_size=0.20, random_state=42, stratify=Y
        )

        self.__model.fit(X_train, y_train)

    def predict(self, args, no_preds):
        preds = dict(zip(self.__model.classes_, self.__model.predict_proba(args))),
        sorted_preds = sorted(
            preds, key=preds.get, reverse=True
        )
        return sorted_preds[:no_preds]
