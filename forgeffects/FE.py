import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

from .reflexive import reflexive
from .GrafoBipartitoEncadenado import GrafoBipartitoEncadenado
from .FEempirical import FEempirical
from .iterative_maxmin_cuadrado import iterative_maxmin_cuadrado
from .process_data import process_data


def FE(CC=None, CE=None, EE=None, causes=None, effects=None, rep=1000, THR=0.5, maxorder=2, device='CPU'):
    # Verificar que CC, CE, EE sean matrices tridimensionales de numpy
    if CC is not None:
        if not isinstance(CC, np.ndarray) or CC.ndim != 3:
            raise ValueError("The 'CC' parameter must be a 3D NumPy array with shape (Number of matrices, row, column).")
    if CE is not None:
        if not isinstance(CE, np.ndarray) or CE.ndim != 3:
            raise ValueError("The 'CE' parameter must be a 3D NumPy array with shape (Number of matrices, row, column).")
    if EE is not None:
        if not isinstance(EE, np.ndarray) or EE.ndim != 3:
            raise ValueError("The 'EE' parameter must be a 3D NumPy array with shape (Number of matrices, row, column).")

    # Selecciona el dispositivo
    if device.upper() == 'GPU' and tf.config.list_physical_devices('GPU'):
        device_name = '/GPU:0'
    else:
        if device.upper() == 'GPU':
            print("No available GPU devices found. Using CPU instead.")
        device_name = '/CPU:0'

    with tf.device(device_name):
        # Conversión de CC, CE, EE a tensores de TensorFlow
        CC = tf.convert_to_tensor(CC, dtype=tf.float32) if CC is not None else None
        CE = tf.convert_to_tensor(CE, dtype=tf.float32) if CE is not None else None
        EE = tf.convert_to_tensor(EE, dtype=tf.float32) if EE is not None else None

        # El resto del código de tu función sin cambios
        provided_names = sum(param is not None for param in [causes, effects])

        if provided_names == 2:
            if CE is None:
                raise ValueError("When 'causes' and 'effects' are provided, CE must exist.")
            if CC is not None and EE is not None:
                if len(causes) != CC.shape[1]:
                    raise ValueError(f"The length of 'causes' must be equal to: {CC.shape[1]}")
                if len(effects) != EE.shape[1]:
                    raise ValueError(f"The length of 'effects' must be equal to: {EE.shape[1]}")
                if CC.shape[1] != CC.shape[2] or EE.shape[1] != EE.shape[2]:
                    raise ValueError("The CC and EE tensors must be square and reflexive.")
                CC = reflexive(CC)
                EE = reflexive(EE)
                tensor = GrafoBipartitoEncadenado(CC, CE, EE)
            else:
                raise ValueError("When 'causes' and 'effects' are provided, CC and EE must exist.")

        elif provided_names == 1:
            if causes is not None and effects is None:
                if CC is None or CE is not None or EE is not None:
                    raise ValueError("When only 'causes' is provided, only CC must exist.")
                if len(causes) != CC.shape[1]:
                    raise ValueError(f"The length of 'causes' must be equal to: {CC.shape[1]}")
                if CC.shape[1] != CC.shape[2]:
                    raise ValueError("The CC tensor must be square and reflexive if CC and EE are not provided.")
                CC = reflexive(CC)
                tensor = CC
            elif effects is not None and causes is None:
                if EE is None or CE is not None or CC is not None:
                    raise ValueError("When only 'effects' is provided, only EE must exist.")
                if len(effects) != EE.shape[1]:
                    raise ValueError(f"The length of 'effects' must be equal to: {EE.shape[1]}")
                if EE.shape[1] != EE.shape[2]:
                    raise ValueError("The EE tensor must be square and reflexive if CC and CE are not provided.")
                EE = reflexive(EE)
                tensor = EE
            else:
                raise ValueError("You must provide either 'causes' or 'effects', not both.")

        elif provided_names == 0:
            if CC is not None and CE is not None and EE is not None:
                if CC.shape[1] != CC.shape[2] or EE.shape[1] != EE.shape[2]:
                    raise ValueError("The CC and EE tensors must be square and reflexive.")
                CC = reflexive(CC)
                EE = reflexive(EE)
                tensor = GrafoBipartitoEncadenado(CC, CE, EE)
            elif CC is not None and CE is not None and EE is None:
                if CC.shape[1] != CC.shape[2]:
                    raise ValueError("The CC tensor must be square and reflexive if EE is not provided.")
                CC = reflexive(CC)
                tensor = GrafoBipartitoEncadenado(CC, CE, EE)
            elif CC is None and CE is not None and EE is not None:
                if EE.shape[1] != EE.shape[2]:
                    raise ValueError("The EE tensor must be square and reflexive if CC is not provided.")
                EE = reflexive(EE)
                tensor = GrafoBipartitoEncadenado(CC, CE, EE)
            elif CC is not None and CE is None and EE is None:
                if CC.shape[1] != CC.shape[2]:
                    raise ValueError("The CC tensor must be square and reflexive if CE and EE are not provided.")
                CC = reflexive(CC)
                tensor = CC
            elif CC is None and CE is None and EE is not None:
                if EE.shape[1] != EE.shape[2]:
                    raise ValueError("The EE tensor must be square and reflexive if CC and CE are not provided.")
                EE = reflexive(EE)
                tensor = EE
            else:
                raise ValueError("You must provide a valid combination of tensors.")
        else:
            raise ValueError("The provided combination of 'causes' and 'effects' is not valid.")

        try:
            tensor_replicas = FEempirical(tensor, rep)
        except tf.errors.ResourceExhaustedError:
            raise ValueError(f"Memory error while creating tensor replicas with rep={rep}.")

        try:
            result_tensors, result_values = iterative_maxmin_cuadrado(tensor_replicas, THR, maxorder)
        except tf.errors.ResourceExhaustedError:
            raise ValueError(f"Memory error while calculating iterative maxmin with rep={rep}.")

        dataframe = []
        for i in range(len(result_tensors)):
            df = process_data(result_tensors[i], result_values[i], CC, CE, EE, causes=causes, effects=effects)
            dataframe.append(df)

    return dataframe
