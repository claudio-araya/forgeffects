import numpy as np
from importlib.resources import files

def load_test_data(filename: str):    
    """
    Loads a .npy file from the package's data directory.

    Args:
    	filename (str): Name of the .npy file to load.

    Returns:
    	numpy.ndarray: Data loaded as a NumPy array.
    """
    
    # Obtiene la ruta del archivo dentro del paquete
    file_path = files("forgeffects.dataset").joinpath(filename)
    return np.load(file_path)

