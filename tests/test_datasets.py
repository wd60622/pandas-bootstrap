import pandas as pd

from bootstrap.datasets import different_mean_and_sigma


def test_different_mean_and_sigma() -> None:
    n = 10
    df = different_mean_and_sigma(n=n, random_state=0)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (n, 2)
    assert df.columns.tolist() == ["group", "x"]


def test_different_mean_and_sigma_with_mu_sigma() -> None:
    n = 1000
    mu = [0, 1]
    sigma = [1, 2]
    df = different_mean_and_sigma(n=n, random_state=0, mu=mu, sigma=sigma)

    mean = df.groupby("group")["x"].mean()
    std = df.groupby("group")["x"].std()

    assert mean[0] < mean[1]
    assert std[0] < std[1]
