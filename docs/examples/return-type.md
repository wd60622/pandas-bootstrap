# Return Type

The return type of `get_samples` is determined by the return type of the `bfunc` function.

| `bfunc` return type | `get_samples` return type | index of `get_samples` |
|---------------------|---------------------------|------------------------|
| `float` or `int`    | `pd.Series`               | sample number |
| `pd.Series`         | `pd.DataFrame`            | sample number | 
| `pd.DataFrame`      | `pd.DataFrame`            | sample number and original index | 

```python
import pandas as pd
import numpy as np

import bootstrap

n_points = 100
df = pd.DataFrame({
    "a": np.random.normal(size=n_points),
    "b": np.random.normal(size=n_points),
    "group": np.random.choice(["A", "B"], size=n_points)
})


def float_return(df: pd.DataFrame, col: str) -> float:
    return df[col].mean()

def series_return(df: pd.DataFrame) -> pd.Series:
    return df.mean()

def df_return(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("group").mean()
    

series = df.boot.get_samples(bfunc=float_return, B=5)
dataframe = df.boot.get_samples(bfunc=series_return, B=5)
dataframe_grouped = df.boot.get_samples(bfunc=df_return, B=5)
```