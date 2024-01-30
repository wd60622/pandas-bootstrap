"""Example datasets to test out the boot attribute."""
from __future__ import annotations

from typing import List, Union

import pandas as pd
import numpy as np


ArrayLike = Union[List, np.ndarray, pd.Series]


def different_mean_and_sigma(n: int, groups: int = 2, random_state: int | None = None, mu: ArrayLike | None = None, sigma: ArrayLike | None =None) -> pd.DataFrame:
    """Generate a dataset with different mean and sigma for each group.

    Args: 
        n: Number of samples
        groups: Number of groups
        random_state: Random state
        mu: Optional mean for each group
        sigma: Optional sigma for each group

    Returns:
        A DataFrame with two columns: group and x

    Examples: 
        Get the confidence intervals for the mean and sigma

        ```python
        import pandas as pd

        from bootstrap.datasets import different_mean_and_sigma

        n = 100
        mu = [1, 2]
        sigma = [1, 4]
        df = different_mean_and_sigma(n=n, random_state=0, mu=mu, sigma=sigma)
        ```
        Which looks like this: 
        ```text
            group         x
        0       1  3.429522
        1       1 -2.833275
        2       1  1.982183
        3       0  1.656475
        4       0 -0.288361
        ..    ...       ...
        95      1  0.564347
        96      0 -0.901635
        97      0  0.891085
        98      0  0.196268
        99      1  6.320654

        [100 rows x 2 columns] 
        ```

        By define our statistic function and pass to `get_samples` method.

        ```python
        def mean_and_sigma_by_group(df: pd.DataFrame) -> pd.DataFrame: 
            return df.groupby("group")["x"].agg(['mean', 'std'])
            
        B = 1_000 
        sample_kwargs = {"random_state": 0}
        df_boot = df.boot.get_samples(
            mean_and_sigma_by_group,
            B=B,
            sample_kwargs=sample_kwargs
        )
        ```
        Which looks like this:

        ```text
                          mean       std 
        group sample
        0     0       1.011128  0.960958
              1       0.995973  0.874010
              2       0.961375  0.941634
              3       0.986562  0.848745
              4       0.749629  0.982982
        ...                ...       ...
        1     995     1.797657  4.103178
              996     2.836222  3.584542
              997     2.587314  3.873845
              998     3.176353  3.444296
              999     2.817353  3.597222

        [2000 rows x 2 columns]
        ```

    """

    rng = np.random.default_rng(random_state)

    mu = mu or rng.normal(0, 1, size=groups)
    sigma = sigma or rng.gamma(1, 1, size=groups)

    mu = np.array(mu)
    sigma = np.array(sigma)

    if len(mu) != groups or len(sigma) != groups:
        raise ValueError(f"mu and sigma must have the same length as groups. {groups = }")

    group = rng.choice(np.arange(groups), size=n)

    return pd.DataFrame({
        'group': group, 
        'x': rng.normal(mu[group], sigma[group], size=n),
    })

    

