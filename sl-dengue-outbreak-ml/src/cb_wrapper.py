"""Small sklearn-compatible wrapper around CatBoost.

CatBoostClassifier modifies its `cat_features` argument internally, which breaks
sklearn.clone() (needed by StackingClassifier / RandomizedSearchCV).  This thin
wrapper keeps plain constructor params and builds the CatBoost model in fit().
It lives in src/ so pickled models can be re-loaded by the front end.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class CatBoostWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, cat_features=None, iterations=500, depth=6, learning_rate=0.05,
                 l2_leaf_reg=3.0, auto_class_weights=None, random_seed=42, thread_count=-1):
        self.cat_features = cat_features
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.auto_class_weights = auto_class_weights
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        X = X.copy()
        cats = list(self.cat_features or [])
        for c in cats:
            X[c] = X[c].astype(str)
        self.model_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg, auto_class_weights=self.auto_class_weights,
            random_seed=self.random_seed, thread_count=self.thread_count,
            cat_features=cats, verbose=0, allow_writing_files=False)
        self.model_.fit(X, np.asarray(y))
        self.classes_ = np.array(self.model_.classes_)
        self.feature_names_in_ = np.array(X.columns)
        return self

    def _prep(self, X):
        X = X[list(self.feature_names_in_)].copy()
        for c in (self.cat_features or []):
            X[c] = X[c].astype(str)
        return X

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prep(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def feature_importances_(self):
        return self.model_.get_feature_importance()
