import tensorflow as tf
import numpy as np

def armar_caminos(tensor1_c, tensor2_c, values, i):
    # Asegurarnos de que 'tensor' y 'values' sean arreglos de NumPy
    if isinstance(tensor1_c, tf.Tensor):
        array1_c = tensor1_c.numpy()
    else:
        array1_c = tensor1_c
    if isinstance(tensor2_c, tf.Tensor):
        array2_c = tensor2_c.numpy()
    else:
        array2_c = tensor2_c
    if isinstance(values, tf.Tensor):
        values_np = values.numpy()
    else:
        values_np = values

    # Extraer las columnas relevantes
    array1 = array1_c[:, [0, 1, i+2]]
    array2 = array2_c[:, [0, 1, 2]]

    # Obtener los valores únicos del primer elemento en ambos arrays
    unique_values_array1 = np.unique(array1[:, 0])
    unique_values_array2 = np.unique(array2[:, 0])

    # Encontrar los valores comunes en el primer elemento
    common_values = np.intersect1d(unique_values_array1, unique_values_array2)

    # Inicializar listas para almacenar los resultados
    paths_list = []
    matched_values_list = []

    # Iterar sobre los valores comunes
    for val in common_values:
        # Filtrar array1 y array2 donde el primer elemento es igual al valor actual
        mask1 = array1[:, 0] == val
        filtered_array1 = array1[mask1]
        filtered_indices1 = np.where(mask1)[0]

        mask2 = array2[:, 0] == val
        filtered_array2 = array2[mask2]
        filtered_indices2 = np.where(mask2)[0]

        # Verificar si hay filas para comparar
        if filtered_array1.shape[0] == 0 or filtered_array2.shape[0] == 0:
            continue  # No hay filas para comparar

        # Expandir dimensiones para comparación vectorizada
        filtered_array1_expanded = filtered_array1[:, np.newaxis, :]  # Shape: (n1_val, 1, 3)
        filtered_array2_expanded = filtered_array2[np.newaxis, :, :]   # Shape: (1, n2_val, 3)

        # Comparar y obtener coincidencias exactas por fila
        comparisons = filtered_array1_expanded == filtered_array2_expanded  # Shape: (n1_val, n2_val, 3)
        matches = np.all(comparisons, axis=2)  # Shape: (n1_val, n2_val)

        # Obtener los índices donde ocurren las coincidencias
        indices = np.argwhere(matches)

        # Si no hay coincidencias, continuar con el siguiente valor
        if indices.shape[0] == 0:
            continue

        # Extraer los índices originales de array1 y array2
        array1_indices = filtered_indices1[indices[:, 0]]
        array2_indices = filtered_indices2[indices[:, 1]]

        # Recolectar las filas que coinciden de los arrays originales
        matched_indices_array1 = array1_c[array1_indices]
        matched_indices_array2 = array2_c[array2_indices]
        matched_values_array2 = values_np[array2_indices]

        # Obtener la columna adicional de array2_c para formar los caminos
        indices_array2_caminos = matched_indices_array2[:, [3]]  

        # Concatenar para formar los caminos
        paths = np.hstack([matched_indices_array1, indices_array2_caminos])

        # Agregar los resultados a las listas
        paths_list.append(paths)
        matched_values_list.append(matched_values_array2)

        del filtered_array1, filtered_array2, filtered_indices1, filtered_indices2, filtered_array1_expanded, filtered_array2_expanded, comparisons, matches, indices, array1_indices, array2_indices, matched_indices_array1, matched_indices_array2, matched_values_array2, indices_array2_caminos

    # Concatenar los resultados de todos los valores comunes
    if paths_list:
        paths = np.vstack(paths_list)
        matched_values = np.concatenate(matched_values_list)
    else:
        # Si no hay coincidencias, devolver arrays vacíos
        paths = np.empty((0, array1_c.shape[1] + 1), dtype=array1_c.dtype)
        matched_values = np.empty((0,), dtype=values_np.dtype)

    return paths, matched_values
