import tensorflow as tf

# @tf.function
def armar_caminos(tensor1_c, tensor2_c, values, i):
    tensor1 = tf.gather(tensor1_c, [0, 1, i+2], axis=1)
    tensor2 = tf.gather(tensor2_c, [0, 1, 2], axis=1)

    # Expandir dimensiones para comparación vectorizada
    tensor1_expanded = tf.expand_dims(tensor1, axis=1)
    tensor2_expanded = tf.expand_dims(tensor2, axis=0) 

    # Comparar y obtener coincidencias exactas por fila
    comparisons = tf.equal(tensor1_expanded, tensor2_expanded)  
    matches = tf.reduce_all(comparisons, axis=2) 

    # Obtener los índices donde ocurren las coincidencias
    indices = tf.where(matches)

    # Extraer los índices de tensor1 y tensor2
    tensor1_indices = indices[:, 0]
    tensor2_indices = indices[:, 1] 

    matched_indices_tensor1 = tf.gather(tensor1_c, tensor1_indices)
    matched_indices_tensor2 = tf.gather(tensor2_c, tensor2_indices)
    matched_values_tensor2 = tf.gather(values, tensor2_indices)

    indices_tensor2_caminos = tf.gather(matched_indices_tensor2, [3], axis=1)

    paths = tf.concat([matched_indices_tensor1, indices_tensor2_caminos], axis=1)

    return paths, matched_values_tensor2
