import os
import unittest
import numpy as np
from forgeffects.FE import FE
from forgeffects.directEffects import directEffects

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

class TestFEFunction(unittest.TestCase):
    
    def test_CC_CE_EE_provided(self):
        """Prueba para el caso en el que se proporcionan CC, CE y EE."""
        result = FE(CC=CC, CE=CE, EE=EE, rep = 10000, THR = 0.5, maxorder = 5)
        self.assertIsInstance(result, list)  # Verifica que el resultado es una lista
        self.assertGreater(len(result), 0)  # Verifica que la lista no está vacía
        #result[0].to_csv('archivo.csv', sep=';', index=False)
        print(result)

    # def test_directEffects(self):
    #     result = directEffects(EE=EE, rep = 1000, THR = 0.5, conf_level=0.95)
    #     print(result)

if __name__ == "__main__":
    unittest.main()
