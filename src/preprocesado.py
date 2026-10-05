import spectral as sp
import scipy.io
import pandas as pd
from chemotools.scatter import StandardNormalVariate
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np

import numpy as np

def snv(cube):
    """
    cube: (H, W, B)
    devuelve: (H, W, B)
    """

    mean = np.mean(cube, axis=1, keepdims=True)
    std = np.std(cube, axis=1, keepdims=True)

    return (cube - mean) / (std + 1e-8)




def savgol(cube, window=15, poly=2, der=0):
    # apply along last axis (bands)
    return savgol_filter(cube, window_length=window, polyorder=poly, der=der, axis=1, mode="nearest")

