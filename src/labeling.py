import numpy as np
from skimage.measure import label, regionprops
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from src.adecuacion import ImagenHyper


def identificar_muestras(img):
    labels = label(img.mask, connectivity=2)
    props = regionprops(labels)

    n_plasticos = len(props)
    print("Plásticos detectados:", n_plasticos)
    return n_plasticos, props


def add_spect(a, name, fig):
    
    fig.add_trace(go.Scatter(
        x=a.WV,
        y=a,
        mode="lines",
        name=name))

def espectros_muestras (props, a):
    fig = go.Figure()
    for k,p in enumerate(props):
        rows = p.coords[:, 0]
        cols = p.coords[:, 1]

        espectros = a[rows, cols]      # (N_pixeles, N_bandas)
        espectro_medio = espectros.mean(axis=0)

        add_spect(espectro_medio, f"Plástico {k+1}")
    fig.show()

def identificar_muestras(mascara, props):
    fig, ax = plt.subplots(figsize=(10, 30))
    ax.imshow(mascara, cmap="gray")

    for k, p in enumerate(props, start=0):
        y, x = p.centroid
        ax.text(x, y, str(k), color="red", fontsize=12)

    plt.show()

def y_prop_label(n):

    y_prop_label = np.zeros(n, dtype=int)
    return y_prop_label

def combinar_labels(imagen, y_prop_label,props):

    ylabel_list = []
    coords = []

    for n, (clase, prop) in enumerate(zip(y_prop_label, props)):
        obj_coords = []

        for y, x in prop.coords:
            ylabel_list.append(clase)
            obj_coords.append([y, x])

        coords.append(np.array(obj_coords))

    imagen.add_label(ylabels=ylabel_list, labelled_coords=coords)
