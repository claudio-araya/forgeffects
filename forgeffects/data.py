import numpy as np
from importlib.resources import files

def load_test_data(filename: str):
    """
    Carga un archivo .npy desde el directorio de datos del paquete.

    Args:
        filename (str): Nombre del archivo .npy a cargar.

    Returns:
        numpy.ndarray: Datos cargados como un array de NumPy.
    """
    # Obtiene la ruta del archivo dentro del paquete
    file_path = files("forgeffects.dataset").joinpath(filename)
    return np.load(file_path)

