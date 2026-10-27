import numpy as np
from skimage.measure import label, regionprops
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from src.adecuacion import ImagenHyper


def identificar_muestras1(img:ImagenHyper):
    """Devuelve el n_plasticos, y los objetos que ha encontrado"""
    # busca huecos en la mascara y las pone en props
    labels = label(img.mask, connectivity=2)
    #Aclaracion de region props 
    props = regionprops(labels)
    # Calcula el numero de huecos
    n_plasticos = len(props)
    print("Plásticos detectados:", n_plasticos)
    return n_plasticos, props


def add_spect(wv, espectro, name, fig):
    """Añade espectro a una figura de seaborn."""
    # Añade espectro a una figura de seaborn
    fig.add_trace(go.Scatter(
        x=wv,
        y=espectro,
        mode="lines",
        name=name
    ))


def espectros_muestras(props, img, preprocesado:str="RAW"):
    """Enseña el espectro medio de cada muestra que ha encontrado en props.
    Preprocesados disponibles SNV.

    """
    #genera figura
    fig = go.Figure()
    #loop de la lista de muestras
    for k, p in enumerate(props):
        # busco columna y filas
        rows = p.coords[:, 0]
        cols = p.coords[:, 1]
        # Busca en la imagen las los espectros de la muestras

        espectros = img.img[rows, cols]
        # promedia el objeto
        espectro_medio = espectros.mean(axis=0)
        # Se activa el preprocesad "RAW" y SNV disponible
        match preprocesado.upper():
            case "RAW":
                espectro = espectro_medio

            case "SNV":
                std = espectro_medio.std()
                if std == 0:
                    espectro = (espectro_medio - espectro_medio.mean()) / 1e-8
                else:
                    espectro = (espectro_medio - espectro_medio.mean()) / std

            case _:
                raise ValueError(f"Preprocesado '{preprocesado}' no reconocido.")
 
        # añade la el propedio de la muestra a la figura de seaborn 
        add_spect(img.WV, espectro, f"Plástico {k}", fig)

    fig.show()

def enseñar_muestras(mascara, props):
    """Genera una figura con el numero de cada muestra. Empezando por 0. Necesita la mascara y el props"""
    #
    fig, ax = plt.subplots(figsize=(10, 30))
    ax.imshow(mascara, cmap="gray")

    for k, p in enumerate(props, start=0):
        y, x = p.centroid
        ax.text(x, y, str(k), color="red", fontsize=12)

    plt.show()

def y_prop_label_generation(n):
    """Genera un array con el mismo numero de muestras para labelear."""
    y_prop_label = np.zeros(n, dtype=int)
    return y_prop_label

def combinar_labels(imagen: "ImagenHyper", y_prop_label: np.array, props: list):
    """Añade labels a la imagen. La muestras de clase 0 seran ignoradas."""
    # genera labels y coordenadas
    coords_list = []
    labels_list = []
    # loops para asignar todos pixeles de la muestra con cada uno de los labes
    for y_prop, prop in zip(y_prop_label, props):
        # ignoramos 0 por ser la clase desconocida
        if y_prop == 0:
            continue
        # copiamos las coordenas 
        coords = prop.coords

        coords_list.append(coords)
        # añadimos 
        labels_list.append(
            np.full(len(coords), y_prop)
        )
    # se pone todo en la lisa y se alabde en la imagen
    npcoords = np.concatenate(coords_list, axis=0)
    ylabel_list = np.concatenate(labels_list, axis=0)
    imagen.add_label(
        ylabels=ylabel_list,
        labelled_coords=npcoords
    )
