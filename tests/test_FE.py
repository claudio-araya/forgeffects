import os
import unittest
import numpy as np
import tensorflow as tf
from forgeffects.FE import FE

# Obtener la ruta de la carpeta actual (donde se encuentra el script)
current_dir = os.path.dirname(__file__)

# Construir la ruta completa hacia los archivos .npy
CC_path = os.path.join(current_dir, 'CC.npy')
CE_path = os.path.join(current_dir, 'CE.npy')
EE_path = os.path.join(current_dir, 'EE.npy')

# Cargar los archivos .npy
CC = np.load(CC_path)
CE = np.load(CE_path)
EE = np.load(EE_path)

# Convertir a tensores de TensorFlow
CC = tf.convert_to_tensor(CC, dtype=tf.float32)
CE = tf.convert_to_tensor(CE, dtype=tf.float32)
EE = tf.convert_to_tensor(EE, dtype=tf.float32)

# Transponer los tensores par que queden de la forma (numero de matrices, filas, columnas)
CC_test = tf.transpose(CC, perm=[2, 0, 1])
CE_test = tf.transpose(CE, perm=[2, 0, 1])
EE_test = tf.transpose(EE, perm=[2, 0, 1])

class TestFEFunction(unittest.TestCase):
    
    def test_CC_CE_EE_provided(self):
        """Prueba para el caso en el que se proporcionan CC, CE y EE."""
        result = FE(CC=CC_test, CE=CE_test, EE=EE_test, rep = 2000, THR = 0.5, maxorder = 10)
        self.assertIsInstance(result, list)  # Verifica que el resultado es una lista
        self.assertGreater(len(result), 0)  # Verifica que la lista no está vacía
        print(result)

if __name__ == "__main__":
    unittest.main()
