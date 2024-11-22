import unittest
import pandas as pd
from forgeffects.FE import FE
from forgeffects.directEffects import directEffects
from forgeffects.data import load_test_data


CC = load_test_data("CC.npy")
CE = load_test_data("CE.npy")
EE = load_test_data("EE.npy")


class TestFunction(unittest.TestCase):
    
    def test_FE_CC_CE_EE_provided(self):
        """Prueba para el caso en el que se proporcionan CC, CE y EE."""
        result = FE(CC=CC, CE=CE, EE=EE, rep=10000, THR=0.5, maxorder=5)
        self.assertIsInstance(result, list)  # Verifica que el resultado es una lista
        self.assertGreater(len(result), 0)  # Verifica que la lista no está vacía

    def test_FE_CC_provided(self):
        """Prueba para el caso en el que se proporciona CC."""
        result = FE(CC=CC, rep=10000, THR=0.5, maxorder=5)
        self.assertIsInstance(result, list)
    
    def test_FE_EE_provided(self):
        """Prueba para el caso en el que se proporciona EE."""
        result = FE(EE=EE, rep=10000, THR=0.5, maxorder=5)
        self.assertIsInstance(result, list)

    def test_directEffects(self):
        """Prueba para la función directEffects."""
        result = directEffects(EE=EE, rep=1000, THR=0.5, conf_level=0.95)
        self.assertIsInstance(result, pd.DataFrame)  # Verifica que el resultado es un DataFrame de pandas


if __name__ == "__main__":
    unittest.main()
