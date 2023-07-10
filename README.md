# Pandas Bootstrap

Bootrapping with Pandas made easy.

## Installation

```bash
pip install pandas-bootstrap
```

## Usage

The module is very easy to use. 

1. `import bootstrap`
2. define statistic function: `def some_func(df: pd.DataFrame | pd.Series):`
3. get bootstrapped samples: `df.boot.get_samples(bfunc=some_func, B=100)`

Below is a simple example of bootstrapping the mean of two columns.

```python
import pandas as pd

import bootstrap

df = pd.DataFrame({
    'a': [1, 2, 3, 4, 5],
    'b': [6, 7, 8, 9, 10],
})

def mean_of_columns(df):
    return df.mean(numeric_only=True)

sample_kwargs = dict(random_state=42)
df_bootstrap = df.boot.get_samples(bfunc=mean_of_columns, B=5, sample_kwargs=sample_kwargs)
```

which results in:

```text 
          a    b
sample          
0       3.0  8.0
1       2.6  7.6
2       4.0  9.0
3       3.2  8.2
4       3.0  8.0
```

## Documentation

Read more in the [documentation](https://wd60622.github.io/pandas-bootstrap/)