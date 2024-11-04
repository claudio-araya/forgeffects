import tensorflow as tf

@tf.function
def maxmin(tensor1, tensor2):
    if tensor1.shape[0] != tensor2.shape[0]:
        raise ValueError("Deben de existir la misma cantidad de expertos en los 2 tensores.")

    if tensor1.shape[2] != tensor2.shape[1]:
        raise ValueError("Las columnas del tensor 1 deben coincidir con las filas del tensor 2.")
    
    # Expando las dimensiones de los tensores para poder hacer la comparación elemento a elemento (fila x columna)
    
    tensor1_expanded = tf.expand_dims(tensor1, axis=3)
    tensor2_expanded = tf.expand_dims(tensor2, axis=1) 

    # Encuentro los mínimos entre los valores de los tensores expandidos

    min_result = tf.minimum(tensor1_expanded, tensor2_expanded) 

    min_result = tf.transpose(min_result, perm=[0, 1, 3, 2])

    # Encuentro el máximo entre los valores
    max_result = tf.reduce_max(min_result, axis=3)

    return min_result, max_result
