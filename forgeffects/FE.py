import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

from .reflexive import reflexive
from .GrafoBipartitoEncadenado import GrafoBipartitoEncadenado
from .FEempirical import FEempirical
from .iterative_maxmin_cuadrado import iterative_maxmin_cuadrado
from .process_data import process_data


def FE(CC=None, CE=None, EE=None, causas=None, efectos=None, rep=1000, THR=0.5, maxorder=2, device='CPU'):
    # Verificar que CC, CE, EE sean matrices tridimensionales de numpy
    if CC is not None:
        if not isinstance(CC, np.ndarray) or CC.ndim != 3:
            raise ValueError("El parámetro 'CC' debe ser una matriz tridimensional de NumPy de la forma (Cantidad de matrices, fila, columna).")
    if CE is not None:
        if not isinstance(CE, np.ndarray) or CE.ndim != 3:
            raise ValueError("El parámetro 'CE' debe ser una matriz tridimensional de NumPy de la forma (Cantidad de matrices, fila, columna).")
    if EE is not None:
        if not isinstance(EE, np.ndarray) or EE.ndim != 3:
            raise ValueError("El parámetro 'EE' debe ser una matriz tridimensional de NumPy de la forma (Cantidad de matrices, fila, columna).")

    # Selecciona el dispositivo
    if device.upper() == 'GPU' and tf.config.list_physical_devices('GPU'):
        device_name = '/GPU:0'
    else:
        if device.upper() == 'GPU':
            print("No se encontraron dispositivos GPU disponibles. Se utilizará la CPU.")
        device_name = '/CPU:0'

    with tf.device(device_name):
        # Conversión de CC, CE, EE a tensores de TensorFlow
        CC = tf.convert_to_tensor(CC, dtype=tf.float32) if CC is not None else None
        CE = tf.convert_to_tensor(CE, dtype=tf.float32) if CE is not None else None
        EE = tf.convert_to_tensor(EE, dtype=tf.float32) if EE is not None else None

        # El resto del código de tu función sin cambios
        provided_names = sum(param is not None for param in [causas, efectos])

        if provided_names == 2:
            if CC is None and EE is None:
                raise ValueError("Cuando 'causas' y 'efectos' se proporcionan, CC y EE deben existir.")
            if CC is not None and EE is not None:
                if len(causas) != CC.shape[1]:
                    raise ValueError(f"La longitud de 'causas' debe ser igual a: {CC.shape[1]}")
                if len(efectos) != EE.shape[1]:
                    raise ValueError(f"La longitud de 'efectos' debe ser igual a: {EE.shape[1]}")
                if CC.shape[1] != CC.shape[2] or EE.shape[1] != EE.shape[2]:
                    raise ValueError("Los tensores CC y EE deben ser cuadrados y reflexivos.")
                CC = reflexive(CC)
                EE = reflexive(EE)
                tensor = GrafoBipartitoEncadenado(CC, CE, EE)
            else:
                raise ValueError("Para 'causas' y 'efectos', CC y EE deben existir.")

        elif provided_names == 1:
            if causas is not None and efectos is None:
                if CC is None or CE is not None or EE is not None:
                    raise ValueError("Cuando solo 'causas' es proporcionado, solo CC debe existir.")
                if len(causas) != CC.shape[1]:
                    raise ValueError(f"La longitud de 'causas' debe ser igual a: {CC.shape[1]}")
                if CC.shape[1] != CC.shape[2]:
                    raise ValueError("El tensor CC debe ser cuadrado y reflexivo si no se proporcionan CC y EE.")
                CC = reflexive(CC)
                tensor = CC
            elif efectos is not None and causas is None:
                if EE is None or CE is not None or CC is not None:
                    raise ValueError("Cuando solo 'efectos' es proporcionado, solo EE debe existir.")
                if len(efectos) != EE.shape[1]:
                    raise ValueError(f"La longitud de 'efectos' debe ser igual a: {EE.shape[1]}")
                if EE.shape[1] != EE.shape[2]:
                    raise ValueError("El tensor EE debe ser cuadrado y reflexivo si no se proporcionan CC y CE.")
                EE = reflexive(EE)
                tensor = EE
            else:
                raise ValueError("Debe proporcionar solo 'causas' o solo 'efectos', no ambos.")

        elif provided_names == 0:
            if CC is not None and CE is not None and EE is not None:
                if CC.shape[1] != CC.shape[2] or EE.shape[1] != EE.shape[2]:
                    raise ValueError("Los tensores CC y EE deben ser cuadrados y reflexivos.")
                CC = reflexive(CC)
                EE = reflexive(EE)
                tensor = GrafoBipartitoEncadenado(CC, CE, EE)
            elif CC is not None and CE is not None and EE is None:
                if CC.shape[1] != CC.shape[2]:
                    raise ValueError("El tensor CC debe ser cuadrado y reflexivo si no se proporciona EE.")
                CC = reflexive(CC)
                tensor = GrafoBipartitoEncadenado(CC, CE, EE)
            elif CC is None and CE is not None and EE is not None:
                if EE.shape[1] != EE.shape[2]:
                    raise ValueError("El tensor EE debe ser cuadrado y reflexivo si no se proporciona CC.")
                EE = reflexive(EE)
                tensor = GrafoBipartitoEncadenado(CC, CE, EE)
            elif CC is not None and CE is None and EE is None:
                if CC.shape[1] != CC.shape[2]:
                    raise ValueError("El tensor CC debe ser cuadrado y reflexivo si no se proporciona CE y EE.")
                CC = reflexive(CC)
                tensor = CC
            elif CC is None and CE is None and EE is not None:
                if EE.shape[1] != EE.shape[2]:
                    raise ValueError("El tensor EE debe ser cuadrado y reflexivo si no se proporciona CC y CE.")
                EE = reflexive(EE)
                tensor = EE
            else:
                raise ValueError("Debe proporcionar una combinación válida de tensores.")
        else:
            raise ValueError("La combinación de 'causas' y 'efectos' proporcionada no es válida.")


        try:
            tensor_replicas = FEempirical(tensor, rep)
        except tf.errors.ResourceExhaustedError:
            raise ValueError(f"Error de memoria al crear tensor_replicas con rep={rep}.")

        try:
            result_tensors, result_values = iterative_maxmin_cuadrado(tensor_replicas, THR, maxorder)
        except tf.errors.ResourceExhaustedError:
            raise ValueError(f"Error de memoria al calcular iterative_maxmin_cuadrado con rep={rep}.")

        dataframe = []
        for i in range(len(result_tensors)):
            df = process_data(result_tensors[i], result_values[i], CC, CE, EE, causas=causas, efectos=efectos)
            dataframe.append(df)

    return dataframe
