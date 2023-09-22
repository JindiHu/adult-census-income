import numpy as np
import pandas as pd

from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, features: pd.DataFrame, target: pd.DataFrame, n_estimators=100, sample_frac=0.7, max_depth=5):
        self.features = features
        self.target = target
        self.n_estimators = n_estimators
        self.sample_frac = sample_frac
        self.max_depth = max_depth
        self.trees = []

    def fit(self):
        # perform bagging
        for _ in range(self.n_estimators):
            # create a bootstrap sample of the data.
            x_sample = self.features.sample(frac=self.sample_frac)
            indices = x_sample.index
            y_sample = self.target.loc[indices]

            # create a decision tree for each bootstrap sample
            dt = DecisionTree(features=x_sample, target=y_sample, max_depth=self.max_depth)
            dt.fit()
            self.trees.append(dt)

    def predict(self, features: pd.DataFrame):
        predictions = []
        for i, _ in features.iterrows():
            # get the predictions from all the decision trees
            cur_row_df = pd.DataFrame(features.loc[i]).transpose()
            tree_predictions = [tree.predict(cur_row_df)[0] for tree in self.trees]

            prediction = np.mean(tree_predictions)
            predictions.append(prediction)
        return predictions

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame):
        actual = list(y_test)
        pred = self.predict(x_test)
        score = 0
        for i in range(len(actual)):
            if pred[i] == actual[i]:
                score = score + 1
        accuracy = score / len(actual)
        print("RandomForest( n_estimators=", self.n_estimators, ", sample_frac=",
              self.sample_frac, ", max_depth=", self.max_depth, ") accuracy:", accuracy)
        print("*" * 10, "\n")
