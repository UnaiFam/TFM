import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def img_rgb(X, r_band=250,g_band=0, b_band=100, out=False):
    """Genera images de Falso RGB"""

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

def imagen_prediccion(y, coords, image_shape, out=False):
    """Genera"""
    H, W = image_shape[:2]
    img = np.zeros((H, W, 3), dtype=np.uint8)

    colors = {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [0, 255, 0],
        3: [0, 0, 255]
    }

    coords = np.asarray(coords)
    y = np.asarray(y)

    if len(coords) != len(y):
        raise ValueError(
            f"coords ({len(coords)}) e ({len(y)}) no tienen el mismo tamaño"
        )

    for (i, j), cls in zip(coords, y):
        img[i, j] = colors.get(cls, [255, 255, 255])

    if out:
        return img

    fig, ax = plt.subplots(figsize=(10, 30))
    ax.imshow(img)
    ax.axis("off")
    
        # -------- LEYENDA --------
    legend_elements = [
        Patch(facecolor=np.array(colors[k]) / 255.0, label=f"Clase {k}")
        for k in colors
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.15, 1)
    )

    plt.show()


def imagen_labelling(props, y_prop, shape, out=False):
        H,W,B=shape
        img = np.zeros((H, W, 3), dtype=np.uint8)
        colors = {
            0: [200, 150, 200],
            1: [255, 0, 0],
            2: [0, 255, 0],
            3: [0, 0, 255]
        }
        for prop, count in zip(props, y_prop):
            coords = prop.coords

                # normaliza a [0, 1]
 
            img[coords[:, 0], coords[:, 1]] = colors[count]


        if out:
            return img

        plt.imshow(img)
        plt.axis("off")
        plt.show()
