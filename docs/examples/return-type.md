---
comments: true
---
# Return Type

The return type of `get_samples` is determined by the return type of the `bfunc` function.

| `bfunc` return type | `get_samples` return type | index of `get_samples` |
|---------------------|---------------------------|------------------------|
| `float` or `int`    | `pd.Series`               | sample number |
| `pd.Series`         | `pd.DataFrame`            | sample number |
| `pd.DataFrame`      | `pd.DataFrame`            | sample number and original index |


## Example

For some concrete examples, we will use the following data:

```python
import pandas as pd
import numpy as np

import bootstrap

n_points = 100
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "a": rng.normal(size=n_points),
    "b": rng.normal(size=n_points),
    "group": rng.choice(["A", "B"], size=n_points)
})

```

### `float` or `int` return a `pd.Series`

```python


def float_return(df: pd.DataFrame, col: str) -> float:
    return df[col].mean()

series = df.boot.get_samples(bfunc=float_return, col="a", B=5)
```

```text
sample
0    0.111948
1    0.157558
2    0.177292
3    0.000501
4    0.082680
Name: stat, dtype: float64
```

### `pd.Series` returns a `pd.DataFrame`


```python
def series_return(df: pd.DataFrame) -> pd.Series:
    return df.mean(numeric_only=True)

dataframe = df.boot.get_samples(bfunc=series_return, B=5)
```

```text
               a         b
sample
0       0.123708 -0.070613
1      -0.011843  0.030508
2       0.027869 -0.006668
3       0.065575  0.043360
4       0.089570  0.125474
```

### `pd.DataFrame` returns a `pd.DataFrame` with another index

```python
def df_return(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("group").mean()

dataframe_grouped = df.boot.get_samples(bfunc=df_return, B=5)
```

```text
                     a         b
group sample
A     0       0.066174 -0.125440
      1      -0.085131 -0.079708
      2       0.128932  0.096542
      3       0.088746 -0.295118
      4      -0.042594  0.063312
B     0      -0.032779  0.035043
      1       0.065471 -0.072222
      2      -0.135504  0.040352
      3       0.120614  0.202248
      4       0.057131  0.027761
```
