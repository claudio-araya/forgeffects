import pandas as pd

def process_direct_results(mx, UCI, boot_pval, causes=None, effects=None, CC=None, CE=None, EE=None, es_cuadrado=False):
    # Mapear números a letras según las condiciones
    mapping = {}

    if causes is not None and effects is None:
        labels = causes
        mapping = {
            "rows": dict(zip(range(len(labels)), labels)),
            "cols": dict(zip(range(len(labels)), labels))
        }
    elif effects is not None and causes is None:
        labels = effects
        mapping = {
            "rows": dict(zip(range(len(labels)), labels)),
            "cols": dict(zip(range(len(labels)), labels))
        }
    elif causes is not None and effects is not None:
        labels_causes = causes
        labels_effects = effects
        mapping = {
            "rows": dict(zip(range(len(labels_causes)), labels_causes)),
            "cols": dict(zip(range(len(labels_effects)), labels_effects))
        }
    else:
        if CC is not None and CE is not None and EE is not None:
            M = CE.shape[1]
            N = CE.shape[2]
            labels_causes = [f'a{i+1}' for i in range(M)]
            labels_effects = [f'b{i+1}' for i in range(N)]
            labels = labels_causes + labels_effects
            mapping = {
                "rows": dict(zip(range(M + N), labels)),
                "cols": dict(zip(range(M + N), labels))
            }
        elif CC is None and CE is not None and EE is None:
            M = CE.shape[1]
            N = CE.shape[2]
            labels_causes = [f'a{i+1}' for i in range(M)]
            labels_effects = [f'b{i+1}' for i in range(N)]
            mapping = {
                "rows": dict(zip(range(M), labels_causes)),
                "cols": dict(zip(range(N), labels_effects))
            }
        elif CC is not None and CE is not None and EE is None:
            M = CE.shape[1]
            N = CE.shape[2]
            labels_causes = [f'a{i+1}' for i in range(M)]
            labels_effects = [f'b{i+1}' for i in range(N)]
            labels = labels_causes + labels_effects
            mapping = {
                "rows": dict(zip(range(M + N), labels)),
                "cols": dict(zip(range(M + N), labels))
            }
        elif CC is None and CE is not None and EE is not None:
            M = CE.shape[1]
            N = CE.shape[2]
            labels_causes = [f'a{i+1}' for i in range(M)]
            labels_effects = [f'b{i+1}' for i in range(N)]
            labels = labels_causes + labels_effects
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
            raise ValueError("The provided combination of tensors is not valid.")

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

