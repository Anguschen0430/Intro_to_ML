# Step 1: Implementing helper functions and core methods in the decision tree class for basic functionality.
import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # Base cases
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        # Stopping conditions
        if depth >= self.max_depth or len(unique_labels) == 1:
            leaf_value = self._calculate_leaf_value(y)
            return {"leaf": True, "value": leaf_value}

        # Find the best split
        feature_index, threshold, feature_is_numeric = self._find_best_split(X, y)
        if feature_index is None:
            leaf_value = self._calculate_leaf_value(y)
            return {"leaf": True, "value": leaf_value}

        # Split the data
        if feature_is_numeric:
            left_mask = X[:, feature_index] <= threshold
        else:
            left_mask = X[:, feature_index] == threshold
        right_mask = ~left_mask
        left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "leaf": False,
            "feature_index": feature_index,
            "threshold": threshold,
            "feature_is_numeric": feature_is_numeric,
            "left": left_subtree,
            "right": right_subtree
        }

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        predictions = [self._predict_tree(x, self.tree) for x in X]
        return np.array(predictions)

    def _predict_tree(self, x, tree_node):
        if tree_node["leaf"]:
            return tree_node["value"]

        if tree_node["feature_is_numeric"]:
            if x[tree_node["feature_index"]] <= tree_node["threshold"]:
                return self._predict_tree(x, tree_node["left"])
            else:
                return self._predict_tree(x, tree_node["right"])
        else:
            if x[tree_node["feature_index"]] == tree_node["threshold"]:
                return self._predict_tree(x, tree_node["left"])
            else:
                return self._predict_tree(x, tree_node["right"])

    def _calculate_leaf_value(self, y):
        return np.bincount(y).argmax()

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_feature_index, best_threshold = None, None
        best_gain = -np.inf
        best_feature_is_numeric = True

        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            if feature_values.dtype == object:
                thresholds = np.unique(feature_values)
                feature_is_numeric = False
            else:
                thresholds = np.unique(feature_values)
                feature_is_numeric = True

            for threshold in thresholds:
                if feature_is_numeric:
                    left_mask = feature_values <= threshold
                else:
                    left_mask = feature_values == threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_feature_is_numeric = feature_is_numeric

        return best_feature_index, best_threshold, best_feature_is_numeric

    def _information_gain(self, parent_y, left_y, right_y):
        weight_left = len(left_y) / len(parent_y)
        weight_right = len(right_y) / len(parent_y)
        gain = self._entropy(parent_y) - (weight_left * self._entropy(left_y) + weight_right * self._entropy(right_y))
        return gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def compute_feature_importances(self, feature_names):
        importances = np.zeros(len(feature_names))

        def recurse(node, depth):
            if node["leaf"]:
                return
            feature_index = node["feature_index"]
            importances[feature_index] += 1
            recurse(node["left"], depth + 1)
            recurse(node["right"], depth + 1)

        recurse(self.tree, 0)
        return importances / np.sum(importances)
