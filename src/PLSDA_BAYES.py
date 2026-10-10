from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score

class PLSDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2, scale=False, max_iter=500):
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.pls = PLSRegression(n_components=n_components, scale=False, max_iter=max_iter)
        self.scaler = StandardScaler() if scale else None
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    
    def transform(self, X):
        pass
    def fit(self, X, y):
        self.classes_=np.unique(y)
        self.encoder_ = LabelBinarizer()
        Y = self.encoder_.fit_transform(y)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if len(self.classes_) == 2 and Y.shape[1] == 1:
            Y = np.column_stack([1 - Y[:, 0], Y[:, 0]])

        self.pls = PLSRegression(
            n_components=self.n_components
        )
        self.pls.fit(X, Y)

        # Predicciones continuas
        yc = self.pls.predict(X)

        self.thr = self.find_thr(X, y)


    def find_thr(self, X, y):
        prob = self.predict_proba(X)
        classes = np.unique(y)

        best_thr = []

        for i, clase in enumerate(classes):
            y_true = (y == clase).astype(int)

            best_score = -1
            best_threshold = 0.5

            for thr in np.arange(0.01, 1.00, 0.01):
                y_pred = (prob[:, i] > thr).astype(int)
                score = f1_score(y_true, y_pred, zero_division=0)

                if score > best_score:
                    best_score = score
                    best_threshold = thr

            best_thr.append(best_threshold)

        return np.array(best_thr)
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    def predict_proba(self, X):
        results=self.pls.predict(X=X)
        return results
    def find_class(yc, class_thr):
        chk_ass = yc > np.asarray(class_thr)

        assigned_class = np.full(yc.shape[0], -1, dtype=int)
        unique = chk_ass.sum(axis=1) == 1
        assigned_class[unique] = np.argmax(chk_ass[unique], axis=1) + 1
        return assigned_class
    def predict(self, X):
        def find_class(yc, class_thr):
            chk_ass = yc > np.asarray(class_thr)

            assigned_class = np.full(yc.shape[0], -1, dtype=int)
            unique = chk_ass.sum(axis=1) == 1
            assigned_class[unique] = np.argmax(chk_ass[unique], axis=1) + 1
            return assigned_class
        probability=self.predict_proba(X)
        thr=self.thr
        prediction=find_class(probability, thr)
        return prediction