import pandas as pd

def process_direct_results(mx, UCI, boot_pval, causas=None, efectos=None, CC=None, CE=None, EE=None, es_cuadrado=False):
    # Mapear números a letras según las condiciones
    mapping = {}

    if causas is not None and efectos is None:
        labels = causas
        mapping = {
            "rows": dict(zip(range(len(labels)), labels)),
            "cols": dict(zip(range(len(labels)), labels))
        }
    elif efectos is not None and causas is None:
        labels = efectos
        mapping = {
            "rows": dict(zip(range(len(labels)), labels)),
            "cols": dict(zip(range(len(labels)), labels))
        }
    elif causas is not None and efectos is not None:
        labels_causas = causas
        labels_efectos = efectos
        mapping = {
            "rows": dict(zip(range(len(labels_causas)), labels_causas)),
            "cols": dict(zip(range(len(labels_efectos)), labels_efectos))
        }
    else:
        if CC is not None and CE is not None and EE is not None:
            M = CE.shape[1]
            N = CE.shape[2]
            labels_causas = [f'a{i+1}' for i in range(M)]
            labels_efectos = [f'b{i+1}' for i in range(N)]
            labels = labels_causas + labels_efectos
            mapping = {
                "rows": dict(zip(range(M + N), labels)),
                "cols": dict(zip(range(M + N), labels))
            }
        elif CC is None and CE is not None and EE is None:
            M = CE.shape[1]
            N = CE.shape[2]
            labels_causas = [f'a{i+1}' for i in range(M)]
            labels_efectos = [f'b{i+1}' for i in range(N)]
            mapping = {
                "rows": dict(zip(range(M), labels_causas)),
                "cols": dict(zip(range(N), labels_efectos))
            }
        elif CC is not None and CE is not None and EE is None:
            M = CE.shape[1]
            N = CE.shape[2]
            labels_causas = [f'a{i+1}' for i in range(M)]
            labels_efectos = [f'b{i+1}' for i in range(N)]
            labels = labels_causas + labels_efectos
            mapping = {
                "rows": dict(zip(range(M + N), labels)),
                "cols": dict(zip(range(M + N), labels))
            }
        elif CC is None and CE is not None and EE is not None:
            M = CE.shape[1]
            N = CE.shape[2]
            labels_causas = [f'a{i+1}' for i in range(M)]
            labels_efectos = [f'b{i+1}' for i in range(N)]
            labels = labels_causas + labels_efectos
            mapping = {
                "rows": dict(zip(range(M + N), labels)),
                "cols": dict(zip(range(M + N), labels))
            }
        elif CC is not None and CE is None and EE is None:
            N = CC.shape[1]
            labels = [f'a{i+1}' for i in range(N)]
            mapping = {
                "rows": dict(zip(range(N), labels)),
                "cols": dict(zip(range(N), labels))
            }
        elif CC is None and CE is None and EE is not None:
            N = EE.shape[1]
            labels = [f'b{i+1}' for i in range(N)]
            mapping = {
                "rows": dict(zip(range(N), labels)),
                "cols": dict(zip(range(N), labels))
            }
        else:
            raise ValueError("La combinación de tensores proporcionados no es válida.")

    num_rows, num_cols = mx.shape
    df = pd.DataFrame({
        'From': [mapping["rows"].get(i, f'unknown{i}') for i in range(num_rows) for _ in range(num_cols)],
        'To': [mapping["cols"].get(j, f'unknown{j}') for _ in range(num_rows) for j in range(num_cols)],
        'Mean': mx.numpy().flatten(),
        'UCI': UCI.numpy().flatten(),
        'p.value': boot_pval.numpy().flatten()
    })

    # Si el tensor es cuadrado, ignorar la diagonal en el DataFrame
    if es_cuadrado:
        df = df[df['From'] != df['To']]

    # Ignorar filas donde el valor de Mean es 0
    df = df[df['Mean'] != 0]

    return df
