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
        result = FE(CC=CC, CE=CE, EE=EE, rep=20000, THR=0.5, maxorder=5)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0) 
        print(result)

    def test_FE_EE_provided_withNames(self):
        """Prueba para el caso en el que se proporcionan CC, CE y EE."""
        result = FE(EE=EE,effects=["effect1","effect2","effect3","effect4"],rep=20000, THR=0.5, maxorder=5)
        self.assertIsInstance(result, list) 
        self.assertGreater(len(result), 0) 
        print(result)

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
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
