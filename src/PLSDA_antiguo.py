import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.special import softmax
 # Example dataset

class PLSDA (BaseEstimator, ClassifierMixin):
    """
    Multiclass PLS-DA wrapper around sklearn's PLSRegression.

    Y is one-hot encoded so PLS regresses X onto a binary indicator matrix.
    Prediction is made by argmax of the predicted Y scores.
    """

    def __init__(self, n_components=2, scale=False, max_iter=500):# Before scale was True
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.pls = PLSRegression(n_components=n_components, scale=False, max_iter=max_iter)
        self.scaler = StandardScaler() if scale else None
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def _one_hot(self, y_encoded, n_classes):
        Y = np.zeros((len(y_encoded), n_classes))
        for i, label in enumerate(y_encoded):
            Y[i, label] = 1
        return Y

    def fit(self, X, y):
        y= np.ravel(y)
        y_enc = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        n_classes = len(self.classes_)

        X_scaled = self.scaler.fit_transform(X) if self.scaler else X
        Y = self._one_hot(y_enc, n_classes)

        self.pls.fit(X_scaled, Y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X) if self.scaler else X
        Y_pred = self.pls.predict(X_scaled)
        idx = np.argmax(Y_pred, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        """Soft scores (not true probabilities, but useful for ranking)."""
        X_scaled = self.scaler.transform(X) if self.scaler else X
        Y_pred = self.pls.predict(X_scaled)
        return softmax(Y_pred, axis=1)
    

    #def predict_proba(self, X): """Soft scores (not true probabilities, but useful for ranking).""" X_scaled = self.scaler.transform(X) if self.scaler else X Y_pred = self.pls.predict(X_scaled) # Shift and normalize to [0,1] range per sample Y_pred -= Y_pred.min(axis=1, keepdims=True) row_sums = Y_pred.sum(axis=1, keepdims=True) row_sums[row_sums == 0] = 1 # avoid div by zero return Y_pred / row_sums

    def transform(self, X):
        """Return X scores (T scores) in latent space."""
        X_scaled = self.scaler.transform(X) if self.scaler else X
        result = self.pls.transform(X_scaled)
        T = result[0] if isinstance(result, tuple) else result
        return T

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def score(self, X, y):
        y = np.ravel(y)
        return np.mean(self.predict(X) == y)