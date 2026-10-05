import numpy as np
import spectral as sp
import pickle



def corrector(ruta_imagen:str,ruta_negro:str,ruta_blanco:str):
    imagen_hiperespectral = sp.open_image(ruta_imagen).load()

    # Cargar el cubo completo en memoria

    imagen_negro =  sp.open_image(ruta_negro).load()                  
    imagen_blanco = sp.open_image(ruta_blanco).load()
   

    imagen_negro_mean=imagen_negro.mean(axis=0)
    imagen_blanco_mean=imagen_blanco.mean(axis=0)

    imagen_corregida=(imagen_hiperespectral-imagen_negro.mean(axis=0))/(imagen_blanco_mean-imagen_negro_mean)
    WV= imagen_hiperespectral.bands.centers

    return imagen_corregida,WV




def img2matrix(img: np.ndarray) -> np.ndarray:
    """
    Convierte imagen (H, W, B) → (H*W, B)
    """
    if img.ndim != 3:
        raise ValueError("La imagen debe ser 3D (H, W, B)")
    
    h, w, b = img.shape
    return img.reshape(h* w, b)

def matrix2img(mat: np.ndarray, h:int, w:int, b:int):  
    """
    Convierte matriz (H*W, B) → (H, W, B)
    """   
    return mat.reshape(h, w, b)


def replace_from_coords(
    img: np.ndarray,
    coords: np.ndarray,
    new_value: int | float = 0
) -> np.ndarray:
    """
    Reemplaza valores en img usando coordenadas [row, col].
    """

    new_img = img.copy()

    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)

    new_img[rows, cols] = new_value

    return new_img

class ImagenHyper:
    """Clase de datos para imagen hiperespectral. 
    imagen en np.ndarray
    Wavelength en np.ndarray
    mascara(opcional) en np.ndarray bool. Si no hay se se genera una con todos los pixeles
    """
    def __init__(self,  img:np.ndarray, WV:np.ndarray, mask=None, ylabels=None):
        self.img = img
        self.shape = img.shape
        self.WV=WV
        self.spectra = img2matrix(img)
        H, W = img.shape[:2]
        self.coords= np.indices((H, W)).reshape(2, -1).T
        if mask is None:
            self.mask = np.ones(img.shape[:2], dtype=bool).astype(bool)
        else:
            self.mask = mask
        self._update_mask_data()

        self.ylabels = ylabels
        self.is_labeled = ylabels is not None
 

    def add_label(self, ylabels:np.array, labelled_coords:np.array):

        self.ylabels = np.asarray(ylabels)
        self.labelled_coords = np.asarray(labelled_coords)
        self.labelled_spectra = self.img[labelled_coords[:, 0], labelled_coords[:, 1], :]
        self.is_labeled = True

    def _update_mask_data(self):
        self.masked_coords = np.argwhere(self.mask)
        self.masked_spectra = self.spectra[self.mask.ravel()]
    def add_mask(self, mask):
        self.mask=np.asarray(mask, bool)
        self._update_mask_data()
    def slicer(self, modo="auto",        fila_ini=None,        fila_fin=None,     col_ini=None,col_fin=None, margin=5):
        if modo == "manual":
            if None in (fila_ini, fila_fin, col_ini, col_fin):
                raise ValueError("modo='manual' requiere límites completos")

        elif modo == "auto":
            filas, cols = np.where(self.mask)

            fila_ini = max(filas.min() - margin, 0)
            fila_fin = min(filas.max() + margin, self.img.shape[0])

            col_ini = max(cols.min() - margin, 0)
            col_fin = min(cols.max() + margin, self.img.shape[1])

        else:
            raise ValueError("modo debe ser 'manual' o 'auto'")

        img_crop = self.img[fila_ini:fila_fin, col_ini:col_fin, :]
        mask_crop = self.mask[fila_ini:fila_fin, col_ini:col_fin]
        new_img = ImagenHyper(
        img=img_crop,
        WV=self.WV,
        mask=mask_crop)

        if self.is_labeled:
            inside = (
                (self.labelled_coords[:, 0] >= fila_ini) &
                (self.labelled_coords[:, 0] < fila_fin) &
                (self.labelled_coords[:, 1] >= col_ini) &
                (self.labelled_coords[:, 1] < col_fin)
            )

            if np.any(inside):

                new_coords = self.labelled_coords[inside].copy()

                # Coordenadas relativas al recorte
                new_coords[:, 0] -= fila_ini
                new_coords[:, 1] -= col_ini

                new_labels = self.ylabels[inside]

                new_img.add_label(
                    ylabels=new_labels,
                    labelled_coords=new_coords
                )

        return new_img

