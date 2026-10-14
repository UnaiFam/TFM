# TFM
Codigo de la trabajo de fin de grado reescrito en python.
Originalmente este proyecto se ecribio en Matlab usando la *classication toolbox*, y se utilizo la gui de *hypertools*. 
El proyecto consistia en el desarrollo y optimización de un flujo de trabajo para la identificación de de imagenes hiperspectrales.
El trabajo usaba imagenes adquiridas de un una camara SWIR de SPECIM. Por tanto generaba las imagenes como una como carpetar se con formato .hdr.
Este proyecto es un port del flujo de trabajo a python usando librerias sklearn, pandas, y numpy.
Originalmente la compresión de las imagenes se hacia despues del preprocesado. Aunque en este repo se hizo antes para acelerar mas el proceso.

La estructura de este proyecto es:
|
|-src
|-Work

### src
|-adecuacion.py

Funciones que contiene:
* **corrector:** Introduciendo las rutas de los de 3 .hdr (imagenes normales, negro, blancos).
* **img2matrix:** Convierte imagen (np.array 2D) (H, W, B) en matriz 3D→en matriz 3D (H*W, B)
* **matrix2img:** Convierte matriz (H*W, B) → (H, W, B)
* **replace_from_coords:** Reemplaza valores de la img en coordenadas np.array([row, col]) usando un np.array(new_value).
* **ImagenHyper *(clase)*:** Esto es una clase cuya funcion es simplificar el manejo de los datos hiperespectral


|-labeling.py
Funciones que contiene:
* **identificar_muestras1:** Devuelve el n_plasticos, y los objetos que ha encontrado
* **add_spect:**Añade figuras a una figura de seaborn.
* **espectros_muestras:**Enseña el espectro medio de cada muestra que ha encontrado en props.
    preprcesados disponibles SNV.
* **enseñar_muestras:** Genera una figura con el numero de cada muestra. Empezando por 0. Necesita la mascara y el props
* **y_prop_label_generation:** Genera un array con el mismo numero de muestras para labelear
* **combinar_labels:** Añade labels a la imagen. La muestras de clase 0 seran ignoradas.
|-PLSDA_BAYES.py
Funciones que contiene:


|-preprocesado.py
Funciones que contiene:
* **snv:**     Finalmente no se utilizo cube: (H, W, B) devuelve: (H, W, B).
* **savgol:**____


|-rgb.py
Funciones que contiene:
* **img_rgb:** Genera images de Falso RGB. Permite cambiar el color cambiando en que vandas se fijas el rgb.
* **imagen_prediccion:** Genera una imagen si se le das las los datos en forma de labes, coodenadas [row, col], y forma. 
* **imagen_labelling:**




|-SIMCA.py
***SIMCA***
Añade el clasificador SIMCA. Utiliza una implementacion ALT SIMCA similar al de *classification-toolbox*.
Funciona como un clasificador one class de sklearn. 
Primero se declara y se elige el numero de componentes del PCA, n de iteraciones y la clase numerica que predice.
    *.fit: Entrena el modelo. Durante el entrenamiento tambien calcula el umbral optimo. Intenta la distancia donde la sensibilidad y especifidad. Si no prioriza sensibiliza.
    *.thr: el umbral de distancia
    *.loadings: enseña los pesos de modelo de PCA
    *.predict_distance: Calcula la distancia media usando la formula $ \sqrt{T hot^2 + q res^2} $ de forma que se combinan las dos distancias de la distancia del modeloy la del centro.
    *.predict: Calcula la distancia media y y compara el umbral.

*** Multi-SIMCA***
Calcula varios modelos ALT-SIMCA en un mismo objeto.
De forma similar al SIMCA hay que declarar el numero de componentes y numero de iteraciones de PCA.

    * .fit:  Entrena todos los modelos
    * .thr: Devuelve lista con los umbrales de clasificacion
    * .predict:
    * .mode: como devuelve las prediccion del modelo. Actualmente:
        * Soft: Devuelve un array de cuyo fila  corresponde a la clase y la columna a la muestra. Esta es la mejor forma de usar la clase ya que permite utilizar al maximo el SIMCA
        * Inter: Devuelve la clase mas cercana usando distancia/umbral. Este es el modo por defecto
        * Hard: Solo predice las muestras en las que solo pertence a una clase. El resto se devuelve como -1.

    * .loadings: devuelve una lista con una lista con los loadings


|-spatial_bin.py
    * **binning_mask:** Realiza una compresión espacial de array booleano  y devuelve: (H//factor, W//factor, B)
    * **binning_hsi:**    Realiza una compresión espacial de cube: (H, W, B) y devuelve: (H//factor, W//factor, B)



