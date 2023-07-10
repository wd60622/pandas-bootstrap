import pandas as pd

import bootstrap  # type: ignore


def dataframe_to_float(df, group: str) -> float:
    return df.groupby("group")["x"].mean().loc[group]


def test_dataframe(df) -> None:
    assert hasattr(df, "boot")

    res = df.boot.get_samples(dataframe_to_float, B=100, group="A")
    assert isinstance(res, pd.Series)
    assert len(res) == 100


def series_to_float(series: pd.Series, head: int) -> float:
    return series.head(head).mean()


def test_series(series) -> None:
    assert hasattr(series, "boot")

    res = series.boot.get_samples(series_to_float, B=100, head=10)
    assert isinstance(res, pd.Series)
    assert len(res) == 100
