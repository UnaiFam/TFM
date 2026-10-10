import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.special import softmax
from sklearn.metrics import recall_score, precision_score



def _find_best_thr(sn,sp,thr_val):
    f= np.where(sn==sp)[0]
    if len(f) > 0:
        # Si hay varios puntos iguales, coger el intermedio
        r = round(len(f) / 2)

        # MATLAB empieza a contar en 1; Python en 0
        f = f[r - 1]

    else:
        # Buscar primer punto donde sensibilidad > especificidad
        f = np.where(sn > sp)[0]

        # Primer índice
        f = f[0]

    # Devolver el threshold correspondiente
    thr = thr_val[f]

    return thr
 

class SIMCA(ClassifierMixin, BaseEstimator):
    """Clasificador Alt-SIMCA para una clase."""

    def __init__(
        self,
        n_components=2,
        max_iter=100,
        _class=1
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self._class = _class

    def fit(self, X, y):

        # Clases que reconoce el clasificador
        self.classes_ = np.unique(y)

        # 1. Seleccionar SOLO la clase modelada
        Xc = X[y == self._class]

        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(Xc)

        # 2. Aplicar PCA a TODAS las muestras
        scores = self.pca.transform(X)

        X_hat = self.pca.inverse_transform(scores)
        self.loadings=self.pca.components_
        E = X - X_hat

        q_res = np.sum(E**2, axis=1)

        eigenvalues = self.pca.explained_variance_

        T_hot = np.sum(
            (scores**2) / eigenvalues,
            axis=1
        )

        # 3. Distancia
        distance = np.sqrt(T_hot**2 + q_res**2)

        # 4. Etiquetas binarias
        class_binary = (y == self._class).astype(int)

        # 5. Buscar threshold
        resthr = _simca_find_thr(
            distance,
            class_binary
        )

        self.thr = resthr["class_thr"]

        return self

    def predict_distance(self, X):

        scores = self.pca.transform(X)

        X_hat = self.pca.inverse_transform(scores)

        E = X - X_hat

        q_res = np.sum(E**2, axis=1)

        eigenvalues = self.pca.explained_variance_

        T_hot = np.sum(
            (scores**2) / eigenvalues,
            axis=1
        )

        distance = np.sqrt(
            T_hot**2 + q_res**2
        )

        return distance

    def predict(self, X):

        prediction = (
            self.predict_distance(X) <= self.thr
        ).astype(int)

        return prediction
    
def _find_best_thr(sn, sp, thr_val):

    f = np.where(sn == sp)[0]

    if len(f) > 0:
        r = (len(f) + 1) // 2
        f = f[r - 1]

    else:
        f = np.where(sn > sp)[0]
        f = f[0]

    return thr_val[f]
def _simca_find_class(D, class_thr):
    return (D <= class_thr).astype(int)

def _calc_class_param_classmodelling(class_calc, class_):

    num_class = 2
    nobj = len(class_)

    conf_mat = np.zeros(
        (num_class, num_class + 1),
        dtype=int
    )

    # -------------------------
    # Modelled class
    # -------------------------

    in_class = np.where(class_ == 1)[0]

    conf_mat[0, 0] = np.sum(class_calc[in_class] == 1)
    conf_mat[0, 1] = np.sum(class_calc[in_class] == 0)
    conf_mat[0, 2] = 0

    # -------------------------
    # Not modelled class
    # -------------------------

    in_class = np.where(class_ == 0)[0]

    conf_mat[1, 0] = np.sum(class_calc[in_class] == 1)
    conf_mat[1, 1] = np.sum(class_calc[in_class] == 0)
    conf_mat[1, 2] = 0

    # -------------------------
    # Metrics
    # -------------------------

    precision = np.zeros(num_class)
    sensitivity = np.zeros(num_class)
    specificity = np.zeros(num_class)

    accuracy = 0

    for g in range(num_class):

        # Precision + sensitivity
        if np.sum(conf_mat[:, g]) > 0:

            precision[g] = (
                conf_mat[g, g]
                / np.sum(conf_mat[:, g])
            )

            sensitivity[g] = (
                conf_mat[g, g]
                / np.sum(conf_mat[g, :num_class])
            )

        else:

            precision[g] = 0
            sensitivity[g] = 0

        # Specificity
        red_rows = [
            k for k in range(num_class)
            if k != g
        ]

        red_mat = conf_mat[
            red_rows,
            :num_class
        ]

        if np.sum(red_mat) > 0:

            specificity[g] = (
                np.sum(red_mat[:, [k for k in range(num_class) if k != g]])
                / np.sum(red_mat)
            )

        else:
            specificity[g] = 0

        accuracy += conf_mat[g, g]

    accuracy /= np.sum(conf_mat[:, :num_class])

    # -------------------------
    # Error rates
    # -------------------------

    not_ass = np.sum(conf_mat[:, -1]) / nobj

    ner = np.mean(sensitivity)

    er = 1 - ner

    return {
        "conf_mat": conf_mat,
        "ner": ner,
        "er": er,
        "accuracy": accuracy,
        "not_ass": not_ass,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity
    }
def _simca_find_thr(D, y):


    dspan = np.max(D)

    thr_val = np.linspace(0, dspan, 101)

    sp = np.zeros(101)
    sn = np.zeros(101)

    for i, thr in enumerate(thr_val):

        class_pred = _simca_find_class(D, thr)

        cp = _calc_class_param_classmodelling(
            class_pred,
            y
        )

        sp[i] = cp["specificity"][0]
        sn[i] = cp["sensitivity"][0]

    class_thr = _find_best_thr(
        sn,
        sp,
        thr_val
    )

    return {
        "sp": sp,
        "sn": sn,
        "thr_val": thr_val,
        "class_thr": class_thr
    }

class SIMCA_Multiclass(ClassifierMixin,BaseEstimator ):
    """Clase para Entrenar varios ALTSIMCA simultaneamente. De momento en caso de que la clase pertenezca a mas de una clase se marca con -1"""
    def __init__(self, n_components=None, max_iter=50, mode="inter"):

        self.n_components = n_components
        self.max_iter = max_iter
        self.mode=mode
        self.models_ = {}
    def fit(self, X, y):

        self.classes_ = np.unique(y)


        for _class in self.classes_:

            if self.n_components is None:
                n_components = 2
            else:
                n_components = self.n_components[_class]

            model = SIMCA(
                n_components=n_components,
                _class=_class,
                max_iter=self.max_iter
            )

            model.fit(X, y)
            print(f"SIMCA for class {_class} trained")
            print(f"with {n_components} components")


            self.models_[_class] = model

        return self


    def predict(self, X):
        """
        Hard devuelve el array con la clase perteneciente
        Soft:Devuelve un array horizontal donde cada fila corresponde a la prediccion de una clase"""
        def soft_prediction(X):

            return  np.array([self.models_[clase].predict(X) for clase in self.classes_])
        match self.mode.lower():
            case "soft":
                return soft_prediction(X).T
            case "hard":
                        predictions = soft_prediction(X)

                        # Número de modelos que aceptan cada muestra
                        n_classes = predictions.sum(axis=0)

                        # Inicialmente, todas las muestras son desconocidas
                        y_pred = np.full(X.shape[0], -1, dtype=int)

                        # Solo se asigna una clase si exactamente un modelo la acepta
                        mask = n_classes == 1

                        y_pred[mask] = self.classes_[predictions[:, mask].argmax(axis=0)]

                        return y_pred

            case "inter":
                prediction =np.array([model.predict_distance(X)/model.thr for model in self.models_.values() ])


                return np.argmin(prediction, axis=0)

            case _:
                raise ValueError(
                    f"Modo '{self.mode.lower()}' no válido. Usa 'soft' o 'hard'."
                )


