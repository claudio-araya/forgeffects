import tensorflow as tf

# @tf.function
def indices(min_result, maxmin_prima, thr):
    
    # Aqui obtengo un tensor unicamente con los valores mayores a thr
    valores_result = tf.boolean_mask(maxmin_prima, maxmin_prima > thr)

    # ----------------------------

    # Máscara para obtener los indices de las coordenadas donde los valores son mayores a thr
    mask_ind = tf.greater(maxmin_prima, thr)
    indices_fe = tf.where(mask_ind)

    # ----------------------------

    # Uso esos índices para obtener unicamente los valores de los minimos donde pasó energía (mayor a thr)
    values = tf.gather_nd(min_result, indices_fe)

    # ----------------------------

    # Valor máximo de los valores donde pasó energía
    max_values = tf.reduce_max(values, axis=1, keepdims=True)

    # Máscara booleana donde se encuentran los valores máximos, para verificar si existe más de un máximo repetido
    max_mask = tf.equal(values, max_values)

    # ----------------------------

    # Genero los índices para la última dimensión
    indices = tf.range(start=0, limit=tf.shape(max_mask)[-1], delta=1)

    # Uso tf.where para reemplazar según la máscara combinada y tener un tensor que indica los índices donde se encuentra el o los máximos
    arg_max_indices = tf.where(max_mask, indices, -1)

    # ----------------------------

    # Después obtengo un tensor plano con los valores distintos de -1
    filtered_tensor = tf.boolean_mask(arg_max_indices, arg_max_indices != -1)

    # ----------------------------

    # Cuento cuantas veces esta repetido un valor maximo por arreglo
    count_non_minus_ones = tf.reduce_sum(tf.cast(max_mask, tf.int32), axis=1)

    # ESTO ES PARA VERIFICAR SOLAMENTE
    #more_than_one = count_non_minus_ones > 1

    # Obtengo un tensor que contiene los índices repetidos
    expanded_indices = tf.repeat(tf.range(tf.shape(indices_fe)[0]), count_non_minus_ones)

    # Obtengo los caminos directos de los efectos encontrados
    result_tensor = tf.gather(indices_fe, expanded_indices)

    # DE MOMENTO LOS TRANSFORMO A INT32 PARA PODER TRABAJARLOS
    result_tensor = tf.cast(result_tensor, dtype=tf.int32)

    # ----------------------------

    # Obtengo los valores mayores a thr pero con las repeticiones de los valores máximos
    result_tensor_values = tf.gather(valores_result, expanded_indices)

    # ----------------------------
    
    # DE MOMENTO TRANSFORMO A INT32 PARA PODER TRABAJARLOS
    filtered_tensor = tf.cast(filtered_tensor, dtype=tf.int32)

    # Expando para poder concatenar
    filtered_tensor = tf.expand_dims(filtered_tensor, axis=1)

    # Concateno los intermedios con los directos
    # Inserto el tensor con los intermedios entre la primera y segunda columna del tensor con los directos
    result_tensor_filtered = tf.concat([result_tensor[:, :2], filtered_tensor, result_tensor[:, 2:]], axis=1)

    del indices_fe, mask_ind,valores_result, values, max_values, max_mask, indices, arg_max_indices, filtered_tensor, count_non_minus_ones, expanded_indices, result_tensor

    return result_tensor_filtered, result_tensor_values
