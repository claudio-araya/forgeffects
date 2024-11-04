import tensorflow as tf
import tensorflow_probability as tfp

# ESTA OTRA FUNCION SOLO PUEDE PROCESAR TENSORES DE LA FORMA (CANTIDAD DE MATRICES, FILA, COLUMNA)

@tf.function
def FEempirical(tensor,rep):

    if rep <= 0:
        raise ValueError('El número de repeticiones debe ser mayor a 0')

    # Genero un tensor con valores aleatorios para usar en los cuantiles

    q = tf.random.uniform([rep], minval=0, maxval=100, dtype=tf.float32)

    # Calculo los cuantiles de los datos

    quantiles = tfp.stats.percentile(tensor, q=q, interpolation='linear', axis=0)

    # Mezclo los cuantiles calculados en los ejes Z

    quantiles = tf.transpose(quantiles, perm=[1, 2, 0])

    shape = quantiles.shape
    tensor_reshaped = tf.reshape(quantiles, (-1, quantiles.shape[-1])) 
    shuffled_tensor = tf.map_fn(tf.random.shuffle, tensor_reshaped)
    shuffled_tensor = tf.reshape(shuffled_tensor, shape)

    return tf.transpose(shuffled_tensor, perm=[2, 0, 1])
