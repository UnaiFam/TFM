import random
from collections import defaultdict
import numpy as np
def combinador_props(*props, split: float = 0.3, seed: int = None):
    """Mezcla aleatoriamente y divide varias listas de props."""
    # genera la lista
    resultado = []
# une todo en la misma lista
    for lista in props:
        resultado.extend(lista)
# randomizer
    rng = random.Random(seed)
    rng.shuffle(resultado)
#separa mas
    n = int(len(resultado) * split)

    return resultado[:n], resultado[n:]



def combinador_props_clase(props, ys, split=0.3, seed=None):
    """
    Mezcla y divide varias listas de props manteniendo
    la proporción de clases de sus respectivos arrays y.
    Tanto props como ys son listas de listas.
    """
    # comprueba si la clases y prop coincide
    if len(props) != len(ys):
        raise ValueError("Debe haber un array y por cada lista de props.")

    #randomizer
    rng = random.Random(seed)
    # genera el diccionaro que va a girar
    clases = defaultdict(list)

    #comprueba si las clases y prop coincide
    for prop, y in zip(props, ys):
        if len(prop) != len(y):
            raise ValueError("Cada prop debe tener tantas etiquetas como elementos.")
        # agrupa props segun clase
        for p, etiqueta in zip(prop, y):
            clases[etiqueta].append(p)
    # genero la listas de entranamiento e y
    train, test = [], []
    y_train, y_test = [], []


    for clase, elementos in clases.items():
        # mezcla los elemento
        rng.shuffle(elementos)
        # genera el split
        n = int(len(elementos) * split)
        # separa laslistas
        test.extend(elementos[:n])
        train.extend(elementos[n:])

        y_test.extend([clase] * n)
        y_train.extend([clase] * (len(elementos) - n))

    # Mezclar ambos conjuntos conservando correspondencia
    train_test = list(zip(train, y_train))
    test_test = list(zip(test, y_test))
#mezlca X
    rng.shuffle(train_test)
    rng.shuffle(test_test)
#separa las propi y labels
    X_train, y_train = zip(*train_test)
    X_test, y_test = zip(*test_test)
# devuelve las array
    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test), np.array(y_test)
    )