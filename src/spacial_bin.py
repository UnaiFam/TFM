from src.adecuacion import ImagenHyper
import numpy as np
from scipy.stats import mode

def binning_mask(mask, factor=2):
    "Funcion que devuelve array booleano reducido"
    H, W = mask.shape

    H2 = (H // factor) * factor
    W2 = (W // factor) * factor

    mask = mask[:H2, :W2]

    blocks = mask.reshape(
        H2 // factor, factor,
        W2 // factor, factor
    )

    return blocks.any(axis=(1, 3))



def binning_hsi(cube, factor=2):
    """
    cube: (H, W, B)
    devuelve: (H//factor, W//factor, B)
    """
    H, W, B = cube.shape

    H2 = (H // factor) * factor
    W2 = (W // factor) * factor

    cube = cube[:H2, :W2]

    return cube.reshape(
        H2 // factor, factor,
        W2 // factor, factor,
        B
    ).mean(axis=(1, 3))

def binning(cube:"ImagenHyper", factor:int=2):
    """ Devuelve una ImagenHyper devuelve: (H//factor, W//factor, B)"""
    cube1=binning_hsi(cube=cube.img,factor=factor)
    mask1=binning_mask(mask=cube.mask, factor=factor)
    return ImagenHyper(img=cube1,mask=mask1 , WV=cube.WV)