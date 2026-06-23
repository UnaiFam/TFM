import matplotlib.pyplot as plt
import numpy as np

def img_rgb(X, r_band=250,g_band=0, b_band=100, out=False):


    rgb = np.stack([
        X[:, :, r_band],
        X[:, :, g_band],
        X[:, :, b_band]
    ], axis=2)

    # Normalizar a 0-1 para matplotlib
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    plt.imshow(rgb)
    plt.title('Imagen RGB aproximada')
    plt.show()
    if out!=False:    return rgb