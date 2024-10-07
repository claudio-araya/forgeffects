from .GrafoBipartitoEncadenado import GrafoBipartitoEncadenado
from .FEempirical import FEempirical
from .iterative_maxmin_cuadrado import iterative_maxmin_cuadrado
from .process_data import process_data

def FE(CC=None, CE=None, EE=None, rep=1000, THR=0.5, maxorder=2):

    if CC is None and CE is None and EE is None:
        raise ValueError("Al menos CE debe ser proporcionado.")
    
    if CC is None and CE is None and EE is not None:
        print("CC None, CE None, EE")
        # if EE.shape[1] != EE.shape[2]:
        #     raise ValueError("El tensor EE debe ser cuadrado si no se proporcionan los tensores CC y CE.")
        # tensor = EE
        raise ValueError("No se puede calcular efectos olvidados con solo el tensor EE.")
        
    
    if CC is not None and CE is None and EE is None:
        print("CC, CE None, EE None")
        # if CC.shape[1] != CC.shape[2]:
        #     raise ValueError("El tensor CC debe ser cuadrado si no se proporcionan los tensores CE y EE.")
        # tensor = CC
        raise ValueError("No se puede calcular efectos olvidados con solo el tensor CC.")

    if CC is None and CE is not None and EE is None:
        print("CC None, CE, EE None")
        if CE.shape[1] != CE.shape[2]:
            raise ValueError("El tensor CE debe ser cuadrado si no se proporcionan los tensores CC y EE.")
        tensor = CE



    if CC is not None and CE is not None and EE is not None:
        print("CC, CE, EE")
        if CC.shape[0] != CE.shape[0] or CE.shape[0] != EE.shape[0]:
            raise ValueError("Los tensores CC, CE y EE deben tener la misma cantidad de matrices.")
        
        tensor, CC, CE, EE = GrafoBipartitoEncadenado(CC, CE, EE)

    elif CC is None and CE is not None and EE is not None:
        print("CC None, CE, EE")
        if CE.shape[0] != EE.shape[0]:
            raise ValueError("Los tensores CE y EE deben tener la misma cantidad de matrices.")
        
        tensor, CC, CE, EE = GrafoBipartitoEncadenado(CC, CE, EE)

    elif CC is not None and CE is not None and EE is None:
        print("CC, CE, EE None")
        if CC.shape[0] != CE.shape[0]:
            raise ValueError("Los tensores CC y CE deben tener la misma cantidad de matrices.")
        
        tensor, CC, CE, EE = GrafoBipartitoEncadenado(CC, CE, EE)

    elif CC is not None and CE is None and EE is not None:
        print("CC, CE None, EE")
        if CC.shape[0] != EE.shape[0]:
            raise ValueError("Los tensores CC y CE deben tener la misma cantidad de matrices.")
        
        tensor, CC, CE, EE = GrafoBipartitoEncadenado(CC, CE, EE)
    

    tensor_replicas = FEempirical(tensor, rep)

    dataframe = []

    result_tensors, result_values = iterative_maxmin_cuadrado(tensor_replicas, THR, maxorder)

    for i in range(len(result_tensors)):
        df = process_data(result_tensors[i], result_values[i], CC, CE, EE)
        dataframe.append(df)

    return dataframe
