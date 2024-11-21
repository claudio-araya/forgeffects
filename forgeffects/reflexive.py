import tensorflow as tf

def reflexive(tensor):

    n_matrices, filas, columnas = tensor.shape
    
    # Crear una máscara de la diagonal principal
    eye = tf.eye(filas, num_columns=columnas, batch_shape=[n_matrices])
    
    # Reemplazar las diagonales con 1
    reflexive_tensor = tensor * (1 - eye) + eye
    
    return reflexive_tensor
