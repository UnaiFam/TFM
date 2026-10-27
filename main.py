import gradio as gr
import os

import pandas as pd
import pickle
import os
import sys
import __main__
import os


import numpy as np
import gradio as gr
import pickle
import src.rgb as rgb
import src.adecuacion as ade
from src.PLSDA_BAYES import PLSDA 
from src.SIMCA import SIMCA



mode = gr.Dropdown(
    choices=["masked", "full"],
    label="Modo",
    value="masked"
)

tipo_modelo = gr.Dropdown(
    choices=["PLSDA", "SIMCA"],
    label="Tipo de modelo",
    value="PLSDA"
)

def imagen(hyperspectral, modelo_path, mode="masked", tipo_modelo="PLSDA"):
    with open(str(hyperspectral), "rb") as f:
        imagen_hyper = pickle.load(f)

    with open(str(modelo_path), "rb") as f:
        modelo = pickle.load(f)

    # Seleccionar espectros y coordenadas
    if mode == "masked":
        X = imagen_hyper.labelled_spectra
        coords = imagen_hyper.labelled_coords
    else:
        X = imagen_hyper.spectra
        coords = imagen_hyper.coords

    prediccion = modelo.predict(X)

    # Dimensiones espaciales de la imagen
    image_shape = imagen_hyper.shape[:2]

    if tipo_modelo == "PLSDA":
        r = rgb.imagen_prediccion(
            y=prediccion,
            image_shape=imagen_hyper.shape[:2],
            coords=coords,
            out=True
        )

        return (
            r,
            np.zeros((*imagen_hyper.shape[:2], 3), dtype=np.uint8),
            np.zeros((*imagen_hyper.shape[:2], 3), dtype=np.uint8)
        )

    elif tipo_modelo == "SIMCA":
        r0 = rgb.imagen_prediccion(
            y=prediccion[:, 0],
            image_shape=image_shape[:2],
            coords=coords,
            out=True
        )
        r1 = rgb.imagen_prediccion(
            y=prediccion[:, 1]*2,
            image_shape=image_shape[:2],
            coords=coords,
            out=True)

        r2 = rgb.imagen_prediccion(
            y=prediccion[:, 1]*3,
            image_shape=image_shape[:2],
            coords=coords,
            out=True
            )
        return r0, r1, r2

demo = gr.Interface(
    fn=imagen,
    inputs=[
        gr.File(type="filepath", label="Archivo hiperespectral"),
        gr.File(type="filepath", label="Modelo entrenado"),
        gr.Radio(
            choices=["masked", "all"],
            value="masked",
            label="Modo de predicción"
        ),
        gr.Radio(
            choices=["PLSDA", "SIMCA"],
            value="PLSDA",
            label="Tipo de modelo"
        )
    ],
    outputs=[
        gr.Image(type="numpy", label="PLS-DA / Clase 1 o SIMCA / PE"),
        gr.Image(type="numpy", label="SIMCA / PP"),
        gr.Image(type="numpy", label="SIMCA / PS")
    ],
    api_name="predict"
)



if __name__ == "__main__":
    demo.launch()
