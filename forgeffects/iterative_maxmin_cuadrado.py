from .maxmin import maxmin
from .indices import indices
from .armar_caminos import armar_caminos

def iterative_maxmin_cuadrado(tensor, thr, order):

    if order <= 1:
        raise ValueError("El orden debe ser mayor a 1.")

    original_tensor = tensor

    gen_tensor = tensor

    result_tensors_list = []
    result_values_list = []

    result_tensors_paths = []
    result_values_paths = []

    for i in range(order-1):
        min_result, maxmin_1_n = maxmin(gen_tensor, original_tensor)
        prima = maxmin_1_n - gen_tensor  # Calcula los efectos de n generación
        result_tensor, result_values = indices(min_result, prima, thr)

        result_tensors_list.append(result_tensor)
        result_values_list.append(result_values)

        if result_values.shape[0] == 0:
            print(f"Solo se encontraron efectos hasta el orden {i+1}")
            break

        if order == 2:
            return result_tensors_list, result_values_list

        if i >= 1:
            if i == 1:
                paths,values = armar_caminos(result_tensors_list[i-1], result_tensors_list[i], result_values_list[i], i)
                result_tensors_paths.append(paths)
                result_values_paths.append(values)
            else:
                paths,values = armar_caminos(result_tensors_paths[i-2], result_tensors_list[i], result_values_list[i], i)
                result_tensors_paths.append(paths)
                result_values_paths.append(values)
        
        gen_tensor = maxmin_1_n 

    result_tensors_paths.insert(0, result_tensors_list[0])
    result_values_paths.insert(0, result_values_list[0])
    
        
    return result_tensors_paths, result_values_paths
