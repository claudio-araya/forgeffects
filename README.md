![PyPI](https://img.shields.io/pypi/v/forgeffects) ![License](https://img.shields.io/pypi/l/forgeffects)

**forgeffects** is a Python package designed for the analysis and computation of forgotten effects and direct effects. It leverages tensor-based computations, aggregating data from multiple key informants to process chained bipartite or complete graphs using 3D tensors.


## Installation

Install the package from PyPI:

```         
pip install forgeffects
```

## Included Datasets

The **forgeffects** package includes three 3D incidence matrices: `CC`, `CE`, and `EE`, used in the study *Application of Forgotten Effects Theory to Evaluate Public Policy on Air Pollution in the Municipality of Valdivia, Chile*.

The data corresponds to sixteen incentives, four behaviors, and ten key informants. These matrices can be loaded using the `load_test_data()` function, which accesses `.npy` files stored in the package.

Example:

```         
import forgeffects as fe

CC = fe.load_test_data("CC.npy")
CE = fe.load_test_data("CE.npy")
EE = fe.load_test_data("EE.npy")
```

# Usage

## Graph Types

- **Complete Graphs**: To perform calculations with complete graphs, you only need to provide either the `CC` matrix (cause-cause relationships) or the `EE` matrix (effect-effect relationships). The `CE` matrix is not required for this type of graph.

- **Chained Bipartite Graphs**: To perform calculations with chained bipartite graphs, all three matrices (`CC`, `CE`, and `EE`) must be provided. These matrices represent cause-cause, cause-effect, and effect-effect relationships, respectively, and are necessary to construct the bipartite graph structure.


## Forgotten Effects

The `FE()` function identifies indirect paths between causes and effects, enabling the discovery of forgotten effects. It uses iterative max-min convolutions to analyze these paths, filtering results based on a specified significance threshold and exploring up to a defined maximum order of forgotten effects.

#### Parameters:

- `CC`: A 3D incidence matrix for cause-cause relationships. This matrix must be in Numpy format with the shape (key informants, rows, columns).
- `CE`: A 3D incidence matrix for cause-effect relationships. This matrix must be in Numpy format with the shape (key informants, rows, columns).
- `EE`: A 3D incidence matrix for effect-effect relationships. This matrix must be in Numpy format with the shape (key informants, rows, columns).
- `causes`: (optional) A list or tuple of strings representing custom names for the causes. If not specified, the causes will be automatically named using the notation $a_1, a_2, \dots, a_n$, where $n$ is the number of evaluated causes.
- `effects`: (optional) A list or tuple of strings representing custom names for the effects. If not specified, the effects will be automatically named using the notation $b_1, b_2, \dots, b_m$, where $m$ is the number of evaluated effects.
- `THR` (float): Defines the degree of truth in which incidence is considered significant within the range [0,1] (default $0.5$).
- `maxorder` (int): Maximum order of forgotten effects to compute (default $2$).
- `reps` (int): Number of replicas for empirical resampling (default $1000$).

#### Returns:

A list of DataFrames, each corresponding to an evaluated order of forgotten effects, with the following columns:

- `From`: Origin node of the indirect relationship.
- `Through_x`: Intermediary nodes (dynamic based on the evaluated order).
- `To`: Destination node of the indirect relationship.
- `Count`: Number of times a forgotten effect is repeated across different experts.
- `Mean`: Mean of the forgotten effects identified.
- `SD`: Standard deviation.

### Example (Chained bipartite graph)

```         
import forgeffects as fe

# Compute forgotten effects
result = fe.FE(CC, CE, EE, rep=10000, THR=0.5, maxorder=3)

# Display the results for the second-order forgotten effects
print(fe_results[0])

# Note: If additional orders are found, the results for third-order effects can be accessed using fe_results[1],
# for fourth-order effects using fe_results[2], and so on.
```

### Results

The example demonstrates the second-order forgotten effects identified by the function

| From | Through1 | To  | Count |   Mean   |    SD    |
|:----:|:--------:|:---:|:-----:|:--------:|:--------:|
| a10  |   a11    | a14 | 2134  | 0.660128 | 0.088254 |
| a13  |   a16    | a1  | 2038  | 0.638836 | 0.080122 |
|  a3  |    a6    | a9  | 1910  | 0.642309 | 0.081263 |
|  a8  |   a13    | a10 | 1905  | 0.713915 | 0.100976 |
| a13  |    a6    | a9  | 1698  | 0.662080 | 0.083397 |
| ...  |   ...    | ... |  ...  |   ...    |   ...    |
|  a9  |    a3    | a10 |   1   | 0.531971 | 0.000000 |
|  a5  |    b2    | b4  |   1   | 0.513954 | 0.000000 |
|  a5  |   a14    | a9  |   1   | 0.541666 | 0.000000 |
|  a2  |   a16    | a5  |   1   | 0.582708 | 0.000000 |
|  a3  |   a10    | b4  |   1   | 0.517474 | 0.000000 |

<p><b>[3725 rows x 6 columns]</b></p>

## Direct Effects

The `directEffects()` function calculates direct effects by estimating mean incidences, confidence intervals, and p-values. It uses a one-sided t-test with bootstrapping to evaluate the significance of direct relationships.

#### Parameters:

- `CC`: A 3D incidence matrix for cause-cause relationships. This matrix must be in Numpy format with the shape (key informants, rows, columns).
- `CE`: A 3D incidence matrix for cause-effect relationships. This matrix must be in Numpy format with the shape (key informants, rows, columns).
- `EE`: A 3D incidence matrix for effect-effect relationships. This matrix must be in Numpy format with the shape (key informants, rows, columns).
- `causes`: (optional) A list or tuple of strings representing custom names for the causes. If not specified, the causes will be automatically named using the notation $a_1, a_2, \dots, a_n$, where $n$ is the number of evaluated causes.
- `effects`: (optional) A list or tuple of strings representing custom names for the effects. If not specified, the effects will be automatically named using the notation $b_1, b_2, \dots, b_m$, where $m$ is the number of evaluated effects.
- `THR` (float): Defines the degree of truth in which incidence is considered significant within the range [0,1] (default $0.5$).
- `conf_level` (float): Confidence level for intervals (default $0.95$).
- `reps` (int): Defines the number of bootstrap replicates (default $10000$).

#### Returns:

A DataFrame with the following columns:

- `From`: Origin node of the incidence.
- `To`: Destination node of the incidence.
- `Mean`: Mean incidence across experts.
- `UCI`: Upper confidence interval limit.
- `p.value`: Calculated p-value.

### Example (Chained bipartite graph)

```         
import forgeffects as fe

# Compute direct effects
de_results = fe.directEffects(CC, CE, EE, rep=10000, THR=0.5, conf_level=0.95)

# Display the results
print(de_results)
```

### Results

The example demonstrates the direct effects identified by the function:

| From |  To  |  Mean  |  UCI   | p.value |
|:----:|:----:|:------:|:------:|:-------:|
|  a1  |  a2  |  0.525 |  0.650 |  0.5979 |
|  a1  |  a3  |  0.450 |  0.590 |  0.2855 |
|  a1  |  a4  |  0.525 |  0.665 |  0.6086 |
|  a1  |  a5  |  0.465 |  0.650 |  0.3648 |
|  a1  |  a6  |  0.645 |  0.805 |  0.8767 |
|  ... |  ... |   ...  |   ...  |   ...   |
|  b3  |  b2  |  0.835 |  0.890 |  0.9996 |
|  b3  |  b4  |  0.645 |  0.785 |  0.8997 |
|  b4  |  b1  |  0.800 |  0.920 |  0.9267 |
|  b4  |  b2  |  0.805 |  0.900 |  0.9974 |
|  b4  |  b3  |  0.820 |  0.900 |  0.9994 |

<p><b>[316 rows x 5 columns]</b></p>


## References

[1] Kaufmann, A. and Gil Aluja, J. *Models for the Research of Forgotten Effects*. Milladoiro, Santiago de Compostela, Spain, 1988.

[2] Manna, E. M., Rojas-Mora, J., & Mondaca-Marino, C. *Application of the Forgotten Effects Theory for Assessing the Public Policy on Air Pollution of the Commune of Valdivia, Chile*. In *From Science to Society* (pp. 61-72). Springer, Cham, 2018.

[3] Mardones-Arias, E.; Rojas-Mora, J. *foRgotten*. R package version 1.1.0, 2022.

[4] Chávez-Bustamante, F.; Mardones-Arias, E.; Rojas-Mora, J.; Tijmes-Ihl, J.\
*A Forgotten Effects Approach to the Analysis of Complex Economic Systems: Identifying Indirect Effects on Trade Networks*. Mathematics, 11(3), Article 531, 2023.

[5] Kohl, M. *MKinfer: Inferential Statistics*. R package version 1.2, 2024.
