import tensorflow as tf

def maxmin(tensor1, tensor2):
    if tensor1.shape[0] != tensor2.shape[0]:
        raise ValueError("There must be the same number of experts in both tensors.")

    if tensor1.shape[2] != tensor2.shape[1]:
        raise ValueError("The columns of tensor1 must match the rows of tensor2.")
    
    # Expando las dimensiones de los tensores para poder hacer la comparación elemento a elemento (fila x columna)
    
    tensor1_expanded = tf.expand_dims(tensor1, axis=3)
    tensor2_expanded = tf.expand_dims(tensor2, axis=1) 

    # Encuentro los mínimos entre los valores de los tensores expandidos

    min_result = tf.minimum(tensor1_expanded, tensor2_expanded) 

    min_result = tf.transpose(min_result, perm=[0, 1, 3, 2])

    # Encuentro el máximo entre los valores
    max_result = tf.reduce_max(min_result, axis=3)

    return min_result, max_result

