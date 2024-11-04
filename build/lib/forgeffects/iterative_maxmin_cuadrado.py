from .maxmin import maxmin
from .indices import indices
from .armar_caminos import armar_caminos

def iterative_maxmin_cuadrado(tensor, thr, order):
    if not (0 <= thr <= 1):
        raise ValueError("El threshold debe estar en los rangos de [0,1].")

    if order <= 1:
        raise ValueError("El orden debe ser mayor a 1.")

    original_tensor = tensor
    gen_tensor = tensor

    result_tensors_list = []
    result_values_list = []

    result_tensors_paths = []
    result_values_paths = []

    for i in range(order - 1):
        # Calcular min_result y maxmin_1_n
        min_result, maxmin_conjugado = maxmin(gen_tensor, original_tensor)
        prima = maxmin_conjugado - gen_tensor  # Efectos de n generación
        
        # Calcular indices con prima y el threshold
        result_tensor, result_values = indices(min_result, prima, thr)

        # Almacenar resultados si no están vacíos
        if result_values.numpy().size == 0:  
            if i == 0:
                raise ValueError(f"No se encontraron efectos con thr {thr}.")
            else:
                print(f"Solo se encontraron efectos hasta el orden {i + 1}")
                break

        result_tensors_list.append(result_tensor)
        result_values_list.append(result_values)

        # Construcción de caminos para órdenes mayores a 1
        if i >= 1:
            previous_paths = result_tensors_paths[-1] if i > 1 else result_tensors_list[0]
            paths, values = armar_caminos(previous_paths, result_tensor, result_values, i)
            result_tensors_paths.append(paths)
            result_values_paths.append(values)

            del previous_paths, paths, values

        # Actualizar gen_tensor para la siguiente iteración
        gen_tensor = maxmin_conjugado

        del min_result, maxmin_conjugado, prima, result_tensor, result_values

    # Agregar el primer resultado a las listas de caminos y valores
    result_tensors_paths.insert(0, result_tensors_list[0])
    result_values_paths.insert(0, result_values_list[0])

    return result_tensors_paths, result_values_paths
