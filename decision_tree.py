import numpy as np
import pandas as pd


def gini_index(y: pd.DataFrame) -> float:
    total = len(y)
    if total == 0:
        return 0.0

    # calculate the class probabilities
    class_probs = np.array([np.sum(y == count) / total for count in y.unique()])

    # calculate the Gini impurity
    gini = 1.0 - np.sum(class_probs ** 2)

    return gini


class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold  # Threshold value for the split
        self.left = left
        self.right = right
        self.value = value  # for leaf nodes in classification


def best_split(x: pd.DataFrame, y: pd.DataFrame):
    num_of_rows = len(x)
    if num_of_rows <= 1:
        return None, None

    df = x.copy()

    # calculate the GINI impurity of the parent node
    parent_gini = gini_index(y)

    # initial max gain to find which split yields the best GINI gain
    max_gain = 0
    best_feature = None
    best_threshold = None

    for feature in df:
        # sort the feature values
        feature_values = df.sort_values(feature)[feature]
        unique_values = feature_values.unique()
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2

        for threshold in thresholds:
            # split the feature based on current threshold
            left_features = df[df[feature] <= threshold]
            right_features = df[df[feature] > threshold]

            # count the number of occurrence after splitting with current threshold
            left_count = len(left_features)
            right_count = len(right_features)

            if left_count == 0 or right_count == 0:
                continue

            left_gini = gini_index(y.loc[left_features.index])
            right_gini = gini_index(y.loc[right_features.index])

            gini_split = (left_count / num_of_rows) * left_gini + (
                    right_count / num_of_rows) * right_gini

            gain = parent_gini - gini_split

            # only update best feature and best threshold if the current gain greater than max gain
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold


class DecisionTree:
    def __init__(self, features, target, max_depth=5):
        self.features = features
        self.target = target
        self.max_depth = max_depth
        self.root = None

    def fit(self):
        self.root = self._build_tree(self.features, self.target, depth=0)
        return

    def _build_tree(self, features, target, depth):
        # stop the recursive function if the depth reach the pre-defined max depth or all target values are identical
        if depth == self.max_depth or len(target.unique()) == 1:
            return DecisionTreeNode(value=np.bincount(target).argmax())

        best_feature, best_threshold = best_split(features, target)

        if best_feature is None:
            return DecisionTreeNode(value=np.bincount(target).argmax())

        # Split the data based on best feature and best threshold
        left_features = features[features[best_feature] <= best_threshold]
        right_features = features[features[best_feature] > best_threshold]
        # construct subtrees recursively
        left_subtree = self._build_tree(left_features, target.loc[left_features.index], depth + 1)
        right_subtree = self._build_tree(right_features, target.loc[right_features.index], depth + 1)

        return DecisionTreeNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def predict(self, features):
        predictions = []
        for _, x in features.iterrows():
            prediction = self._traverse_tree(self.root, x)
            predictions.append(prediction)
        return predictions

    def _traverse_tree(self, node, x):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(node.left, x)
        else:
            return self._traverse_tree(node.right, x)

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame):
        actual = list(y_test)
        pred = self.predict(x_test)
        score = 0
        for i in range(len(actual)):
            if pred[i] == actual[i]:
                score = score + 1
        accuracy = score / len(actual)
        print("DecisionTree( max_depth=", self.max_depth, ") accuracy:", accuracy)
        print("*" * 10, "\n")
