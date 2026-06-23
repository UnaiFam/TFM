import spectral as sp
import scipy.io
import pandas as pd
from chemotools.scatter import StandardNormalVariate
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np

def snv(cube):
    # cube: (H, W, B)
    mean = np.mean(cube, axis=2, keepdims=True)
    std = np.std(cube, axis=2, keepdims=True)
    if std==0:
        return (cube - mean) / (std+ 1e-8)
    else:
        return (cube - mean) / (std)



def savgol(cube, window=15, poly=2, der=0):
    # apply along last axis (bands)
    return savgol_filter(cube, window_length=window, polyorder=poly, der=der, axis=2, mode="nearest")
