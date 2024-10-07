import numpy as np
import pandas as pd

def process_data(tensor, values, CC=None, CE=None, EE=None):

    tensor = tensor[:, 1:]

    # Obtener el número de columnas en el tensor
    n_cols = tensor.shape[1]
    
    # Generar nombres de columnas dinámicamente
    col_names = ['From'] + [f'Through{i+1}' for i in range(n_cols - 2)] + ['To']
    
    # Convertir el tensor y los valores a arreglos numpy
    tensor_np = tensor.numpy()
    values_np = values.numpy()
    
    # Crear un diccionario para almacenar las filas y sus posiciones
    row_dict = {}
    for idx, row in enumerate(tensor_np):
        row_key = tuple(row.tolist())
        if row_key in row_dict:
            row_dict[row_key].append(idx)
        else:
            row_dict[row_key] = [idx]
    
    # Preparar datos para construir el DataFrame
    data = []
    for row_key, positions in row_dict.items():
        # Convertir posiciones a arreglo numpy
        positions_np = np.array(positions)
        # Extraer los valores correspondientes a estas posiciones
        values_at_positions = values_np[positions_np]
        count = len(positions)
        mean_value = values_at_positions.mean()
        std_value = values_at_positions.std(ddof=0)  # Desviación estándar
        # Construir la fila de datos
        data_row = list(row_key) + [count, mean_value, std_value]
        data.append(data_row)
    
    # Construir los nombres de las columnas
    df_columns = col_names + ['Count', 'Mean', 'SD']
    
    # Crear el DataFrame
    df = pd.DataFrame(data, columns=df_columns)
    
    # Ordenar el DataFrame por 'Count' de mayor a menor
    df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    
    # Mapear números a letras según las condiciones
    mapping = {}
    if (CC is None) and (EE is None):
        # Caso 1: Solo CE existe
        if CE is not None:
            N = CE.shape[1]
            labels = [f'I{i+1}' for i in range(N)]
            mapping = dict(zip(range(N), labels))
    else:
        # Caso 2: CC o EE existen
        M = CC.shape[1]  
        N = EE.shape[2]
        labels_a = [f'a{i+1}' for i in range(M)]
        labels_b = [f'b{i+1}' for i in range(N)]
        mapping = dict(zip(range(M + N), labels_a + labels_b))
    
    # Aplicar el mapeo al DataFrame
    for col in col_names:
        df[col] = df[col].map(mapping)
    
    return df
