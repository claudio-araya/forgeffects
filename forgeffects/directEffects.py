import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from .GrafoBipartitoEncadenado import GrafoBipartitoEncadenado
from .process_direct_results import process_direct_results

def directEffects(CC=None, CE=None, EE=None, causes=None, effects=None, rep=1000, THR=0.5, conf_level=0.95):

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
        
    # Verificar que 'causes' y 'effects' sean arrays con elementos de tipo string
    if causes is not None:
        if not isinstance(causes, (list, tuple)) or not all(isinstance(item, str) for item in causes):
            raise ValueError("The 'causes' parameter must be an array (list or tuple) of strings.")
    if effects is not None:
        if not isinstance(effects, (list, tuple)) or not all(isinstance(item, str) for item in effects):
            raise ValueError("The 'effects' parameter must be an array (list or tuple) of strings.")

    if conf_level < 0 or conf_level > 1:
        raise ValueError("The confidence level must be in the range [0,1].")

    CC = tf.convert_to_tensor(CC, dtype=tf.float32) if CC is not None else None
    CE = tf.convert_to_tensor(CE, dtype=tf.float32) if CE is not None else None
    EE = tf.convert_to_tensor(EE, dtype=tf.float32) if EE is not None else None

    # Verificar la cantidad de nombres proporcionados
    provided_names = sum(param is not None for param in [causes, effects])

    # Verificación según el número de nombres proporcionados
    if provided_names == 2:
        if CE is None:
            raise ValueError("When 'causes' and 'effects' are provided, CE must exist.")
        if CC is not None and EE is not None:
            if len(causes) != CC.shape[1]:
                raise ValueError(f"The length of 'causes' must be equal to: {CC.shape[1]}")
            if len(effects) != EE.shape[1]:
                raise ValueError(f"The length of 'effects' must be equal to: {EE.shape[1]}")
            tensor = GrafoBipartitoEncadenado(CC, CE, EE)
        if CC is None and EE is None:
            tensor = CE
        else:
            raise ValueError("Missing CC, EE, or both.")

    elif provided_names == 1:
        if causes is not None and effects is None:
            if CC is None or CE is not None or EE is not None:
                raise ValueError("When only 'causes' is provided, only CC must exist.")
            if len(causes) != CC.shape[1]:
                raise ValueError(f"The length of 'causes' must be equal to: {CC.shape[1]}")
            tensor = CC
        elif effects is not None and causes is None:
            if EE is None or CE is not None or CC is not None:
                raise ValueError("When only 'effects' is provided, only EE must exist.")
            if len(effects) != EE.shape[1]:
                raise ValueError(f"The length of 'effects' must be equal to: {EE.shape[1]}")
            tensor = EE
        else:
            raise ValueError("You must provide either 'causes' or 'effects', not both if only one tensor exists.")

    elif provided_names == 0:
        if CC is not None and CE is not None and EE is not None:
            tensor = GrafoBipartitoEncadenado(CC, CE, EE)
        elif CC is not None and CE is None and EE is None:
            tensor = CC
        elif CC is None and CE is None and EE is not None:
            tensor = EE
        elif CC is None and CE is not None and EE is None:
            tensor = CE
        else:
            raise ValueError("You must provide a valid combination of tensors.")

    # Calcula la media a lo largo del eje de matrices (primero)
    mx = tf.reduce_mean(tensor, axis=0)

    # Define nx como el número de matrices
    nx = tf.cast(tf.shape(tensor)[0], tf.float32)

    # Centrar los datos restando la media a cada réplica
    x_cent = tensor - mx[tf.newaxis, ...]

    num_samples = tf.shape(tensor)[0]  # Número de matrices

    # Índices aleatorios para las réplicas bootstrap
    indices = tf.random.uniform([rep, num_samples], minval=0, maxval=num_samples, dtype=tf.int32)

    # Muestreo del tensor centrado usando los índices generados
    X = tf.gather(x_cent, indices, axis=0)
    
    # Calcula la media de las muestras bootstrap a lo largo del primer eje (réplicas)
    MX = tf.reduce_mean(X, axis=1)

    # Error estándar de las medias remuestreadas
    VX = tf.reduce_sum(tf.square(X - MX[:, tf.newaxis, ...]), axis=1) / (nx - 1)
    STDERR = tf.sqrt(VX / nx)

    # Estadístico t para las réplicas bootstrap
    TSTAT = MX / STDERR

    # Estadístico t de la muestra original
    original_var = tf.math.reduce_variance(tensor, axis=0)
    tstat = (mx - THR) / tf.sqrt(original_var / nx)

    EFF = MX + mx

    conf_level = conf_level * 100

    UCI = tfp.stats.percentile(EFF, q=conf_level, interpolation='linear', axis=0)

    # Cálculo del valor p bootstrap
    boot_pval = tf.maximum(tf.reduce_mean(tf.cast(TSTAT < tstat[tf.newaxis, ...], tf.float32), axis=0), 1 / rep)

    # Verificación de si el tensor es cuadrado
    es_cuadrado = mx.shape[0] == mx.shape[1]

    # Construcción del DataFrame final
    df_resultados = process_direct_results(mx, UCI, boot_pval, causes, effects, CC, CE, EE, es_cuadrado)
    
    return df_resultados

