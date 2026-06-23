import numpy as np
import spectral as sp




def corrector(ruta_imagen,ruta_negro,ruta_blanco):
    imagen_hiperespectral = sp.open_image(ruta_imagen).load()

    # Cargar el cubo completo en memoria

    imagen_negro =  sp.open_image(ruta_negro).load()                  
    imagen_blanco = sp.open_image(ruta_blanco).load()
   
    print(imagen_hiperespectral.shape,imagen_blanco.shape, imagen_blanco.shape )

    im_largo, im_ancho, im_bandas=imagen_hiperespectral.shape


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

    def __init__(self, name:str, img:np.array, WV, mask=None, ylabels=None):

        self.name = name
        self.img = img
        self.shape = img.shape
        self.WV=WV
        self.spectra = img2matrix(img)
        H, W = img.shape[:2]
        self.coords = np.indices((H, W)).reshape(2, -1).T

        if mask is None:
            self.mask = np.ones(img.shape[:2], dtype=bool).astype(bool)
        else:
            self.mask = mask

        self.ylabels = ylabels
        

        self.spectra_coords= np.argwhere(self.img!=False)
        self.masked_coords =np.argwhere(self.mask)
        self.masked_spectra=img[self.mask]


        
        if self.ylabels!=None:
            self.ylabels= False
        else:
            self.ylabels= True



    def add_label(self, ylabels, labelled_coords):
        self.ylabels = ylabels 
        self.labelled_coords=labelled_coords

        self.labelled_spectra = []

        for coords in labelled_coords:
            spectra_obj = []

            for y, x in coords:
                # caso imagen H x W x bands
                spectra_obj.append(self.img[y, x, :])

        self.labelled_spectra.append(np.array(spectra_obj))
        

    def add_plastics(self):
        pass




    def add_mask(self, mask):

        # asegurar forma correcta
        if mask.shape != self.img.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} != image shape {self.img.shape[:2]}"
            )

        # forzar booleano limpio
        mask = mask.astype(bool)

        self.mask = mask
        self.masked_coords = np.argwhere(mask)

        # ESTO ES CORRECTO SOLO SI mask es (H,W)
        self.masked_spectra = self.img[mask]

    

        

    def slicer(
        self,
        modo="auto",
        fila_ini=None,
        fila_fin=None,
        col_ini=None,
        col_fin=None,
    ):

        if modo == "manual":

            img_crop = self.img[fila_ini:fila_fin,
                                col_ini:col_fin, :]

            mask_crop = self.mask[fila_ini:fila_fin,
                                  col_ini:col_fin]

        elif modo == "auto":

            filas, cols = np.where(self.mask)

            margin = 5

            fila_ini = max(filas.min() - margin, 0)
            fila_fin = min(filas.max() + margin, self.img.shape[0])

            col_ini = max(cols.min() - margin, 0)
            col_fin = min(cols.max() + margin, self.img.shape[1])

            img_crop = self.img[fila_ini:fila_fin,
                                col_ini:col_fin, :]

            mask_crop = self.mask[fila_ini:fila_fin,
                                  col_ini:col_fin]

        else:
            raise ValueError("modo debe ser 'manual' o 'auto'")

        return ImagenHyper(
            name=f"{self.name}_crop",
            img=img_crop,
            WV=self.WV,
            mask=mask_crop,
            ylabels=self.ylabels
        )
    

