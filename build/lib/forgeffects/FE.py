import tensorflow as tf
import numpy as np

from .GrafoBipartitoEncadenado import GrafoBipartitoEncadenado
from .FEempirical import FEempirical
from .iterative_maxmin_cuadrado import iterative_maxmin_cuadrado
from .process_data import process_data

def FE(CC=None, CE=None, EE=None, causas=None, efectos=None, rep=1000, THR=0.5, maxorder=2):

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

    # Verificar que 'causas' y 'efectos' sean listas de strings
    if causas is not None:
        if not isinstance(causas, list) or not all(isinstance(c, str) for c in causas):
            raise ValueError("El parámetro 'causas' debe ser una lista de strings.")
    if efectos is not None:
        if not isinstance(efectos, list) or not all(isinstance(e, str) for e in efectos):
            raise ValueError("El parámetro 'efectos' debe ser una lista de strings.")

    CC = tf.convert_to_tensor(CC, dtype=tf.float32) if CC is not None else None
    CE = tf.convert_to_tensor(CE, dtype=tf.float32) if CE is not None else None
    EE = tf.convert_to_tensor(EE, dtype=tf.float32) if EE is not None else None

    provided_names = sum(param is not None for param in [causas, efectos])

    if provided_names == 2:
        if CC is None and EE is None:
            raise ValueError("Cuando 'causas' y 'efectos' se proporcionan, CC y EE deben existir.")
        if CC is not None and EE is not None:
            if len(causas) != CC.shape[1]:
                raise ValueError(f"La longitud de 'causas' debe ser igual a: {CC.shape[1]}")
            if len(efectos) != EE.shape[1]:
                raise ValueError(f"La longitud de 'efectos' debe ser igual a: {EE.shape[1]}")
            tensor = GrafoBipartitoEncadenado(CC, CE, EE)
        else:
            raise ValueError("Para 'causas' y 'efectos', CC y EE deben existir.")

    elif provided_names == 1:
        if causas is not None and efectos is None:
            if CC is None or CE is not None or EE is not None:
                raise ValueError("Cuando solo 'causas' es proporcionado, solo CC debe existir.")
            if CC.shape[1] != CC.shape[2]:
                raise ValueError("El tensor CC debe ser cuadrado y reflexivo si no se proporcionan CC y EE.")
            if len(causas) != CC.shape[1]:
                raise ValueError(f"La longitud de 'causas' debe ser igual a: {CC.shape[1]}")
            tensor = CC
        elif efectos is not None and causas is None:
            if EE is None or CE is not None or CC is not None:
                raise ValueError("Cuando solo 'efectos' es proporcionado, solo EE debe existir.")
            if EE.shape[1] != EE.shape[2]:
                raise ValueError("El tensor EE debe ser cuadrado y reflexivo si no se proporcionan CC y CE.")
            if len(efectos) != EE.shape[1]:
                raise ValueError(f"La longitud de 'efectos' debe ser igual a: {EE.shape[1]}")
            tensor = EE
        else:
            raise ValueError("Debe proporcionar solo 'causas' o solo 'efectos', no ambos.")

    elif provided_names == 0:
        if CC is not None and CE is not None and EE is not None:
            tensor = GrafoBipartitoEncadenado(CC, CE, EE)
        elif CC is not None and CE is not None and EE is None:
            tensor = GrafoBipartitoEncadenado(CC, CE, EE)
        elif CC is None and CE is not None and EE is not None:
            tensor = GrafoBipartitoEncadenado(CC, CE, EE)
        elif CC is not None and CE is None and EE is None:
            tensor = CC
        elif CC is None and CE is None and EE is not None:
            tensor = EE
        else:
            raise ValueError("Debe proporcionar una combinación válida de tensores.")
    else:
        raise ValueError("La combinación de 'causas' y 'efectos' proporcionada no es válida.")

    # Intentar reducir rep en caso de error de memoria al crear tensor_replicas
    while True:
        try:
            tensor_replicas = FEempirical(tensor, rep)
            break
        except tf.errors.ResourceExhaustedError:
            print(f"Error de memoria con rep={rep}. Reduciendo rep a la mitad.")
            rep = max(1, rep // 2)  # Reducir rep a la mitad, pero no menos de 1
            if rep == 1:
                raise RuntimeError("No se puede reducir más el número de réplicas para FEempirical.")

    # Intentar reducir rep en caso de error de memoria en iterative_maxmin_cuadrado
    while True:
        try:
            result_tensors, result_values = iterative_maxmin_cuadrado(tensor_replicas, THR, maxorder)
            break
        except tf.errors.ResourceExhaustedError:
            print(f"Error de memoria al calcular iterative_maxmin_cuadrado con rep={rep}. Reduciendo rep.")
            rep = max(1, rep // 2)  # Reducir rep a la mitad, pero no menos de 1
            if rep == 1:
                raise RuntimeError("No se puede reducir más el número de réplicas para iterative_maxmin_cuadrado.")
            # Rehacer las réplicas con el nuevo valor de rep
            tensor_replicas = FEempirical(tensor, rep)

    dataframe = []
    for i in range(len(result_tensors)):
        df = process_data(result_tensors[i], result_values[i], CC, CE, EE, causas=causas, efectos=efectos)
        dataframe.append(df)

    return dataframe
