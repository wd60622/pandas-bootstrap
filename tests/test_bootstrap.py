from typing import Dict

from joblib import Parallel

import pytest

import pandas as pd

from bootstrap.bootstrap import (
    bootstrap,
    UnsupportedReturnType,
)


# DataFrame functions
def dataframe_to_dataframe_with_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("group").mean()


def dataframe_to_dataframe_with_index_inferred(df):
    return dataframe_to_dataframe_with_index(df)


def dataframe_to_series_with_index(df: pd.DataFrame) -> pd.Series:
    return df.groupby("group").mean()["y"]


def dataframe_to_series_with_index_inferred(df):
    return dataframe_to_series_with_index(df)


def dataframe_to_series(df: pd.DataFrame) -> pd.Series:
    return df.mean(numeric_only=True)


def dataframe_to_series_inferred(df):
    return dataframe_to_series(df)


def dataframe_to_float(df: pd.DataFrame) -> float:
    return df.mean(numeric_only=True).mean()


def dataframe_to_float_inferred(df):
    return dataframe_to_float(df)


dataframe_lambda_func = lambda df: df.mean(numeric_only=True)  # noqa


# Series functions
def series_to_float(series: pd.Series) -> float:
    return series.mean()


def series_to_float_inferred(series):
    return series_to_float(series)


def series_to_int(series: pd.Series) -> int:
    return int(series.mean())


def series_to_int_inferred(series):
    return series_to_int(series)


def series_to_dict(series: pd.Series) -> Dict[str, float]:
    return dict(mean=series.mean(), std=series.std())


def series_to_series(series: pd.Series) -> pd.Series:
    return pd.Series(
        {
            "mean": series.mean(),
            "std": series.std(),
        }
    )


def series_to_series_inferred(series):
    return series_to_series(series)


series_lambda_func = lambda ser: ser.mean()  # noqa


@pytest.mark.parametrize("parallel", [None, Parallel(n_jobs=2)])
@pytest.mark.parametrize(
    "bfunc, n_rows_multiplier, index_type, return_type",
    [
        (dataframe_to_dataframe_with_index, 2, pd.MultiIndex, pd.DataFrame),
        (dataframe_to_dataframe_with_index_inferred, 2, pd.MultiIndex, pd.DataFrame),
        (dataframe_to_series, 1, pd.Index, pd.DataFrame),
        (dataframe_to_series_inferred, 1, pd.Index, pd.DataFrame),
        # (dataframe_to_series_with_index, 1000, pd.MultiIndex, pd.DataFrame),
        (dataframe_to_float, 1, pd.Index, pd.Series),
        (dataframe_to_float_inferred, 1, pd.Index, pd.Series),
        (dataframe_lambda_func, 1, pd.Index, pd.DataFrame),
    ],
)
def test_bootstrap_dataframe(
    df, parallel, bfunc, n_rows_multiplier, index_type, return_type
) -> None:
    B = 50
    df_boot = df.pipe(bootstrap, bfunc, B=B, parallel=parallel)

    assert isinstance(df_boot, return_type)
    assert df_boot.shape[0] == (B * n_rows_multiplier)
    assert isinstance(df_boot.index, index_type)


@pytest.mark.parametrize("parallel", [None, Parallel(n_jobs=2)])
@pytest.mark.parametrize(
    "bfunc, return_type",
    [
        (series_to_float, pd.Series),
        (series_to_float_inferred, pd.Series),
        (series_to_int, pd.Series),
        (series_to_int_inferred, pd.Series),
        # series_to_dict,
        (series_to_series, pd.DataFrame),
        # (series_to_dict, pd.DataFrame),
        (series_to_series_inferred, pd.DataFrame),
        (series_lambda_func, pd.Series),
    ],
)
def test_bootstrap_series(series, parallel, bfunc, return_type) -> None:
    B = 50
    series_boot = series.pipe(bootstrap, bfunc, B=B, parallel=parallel)

    assert isinstance(series_boot, return_type)
    assert series_boot.shape[0] == B
    assert isinstance(series_boot.index, pd.Index)


def function_with_kwargs_to_float(df: pd.DataFrame, col: str) -> float:
    return df[col].mean()


def function_with_kwargs_to_float_inferred(df, col: str):
    return function_with_kwargs_to_float(df, col)


@pytest.mark.parametrize("parallel", [None, Parallel(n_jobs=2)])
@pytest.mark.parametrize(
    "bfunc, kwargs",
    [
        (function_with_kwargs_to_float, {"col": "x"}),
        (function_with_kwargs_to_float_inferred, {"col": "x"}),
    ],
)
def test_support_for_kwargs(df, parallel, bfunc, kwargs) -> None:
    B = 50
    series_boot = df.pipe(bootstrap, bfunc=bfunc, B=B, parallel=parallel, **kwargs)

    assert isinstance(series_boot, pd.Series)
    assert series_boot.shape[0] == B
    assert isinstance(series_boot.index, pd.Index)


def unsupported_return_type_str(df) -> str:
    return "Hello World"


def unsupported_return_type_bool(df) -> bool:
    return True


def unsupported_return_type_list(df) -> list:
    return [1, 2, 3]


def unsupported_return_type_tuple(df) -> tuple:
    return (1, 2, 3)


def unsupported_return_type_set(df) -> set:
    return {1, 2, 3}


@pytest.mark.parametrize(
    "parallel", [None, Parallel(n_jobs=2), Parallel(n_jobs=2, prefer="threads")]
)
@pytest.mark.parametrize(
    "data_input", [pd.Series(dtype=float), pd.DataFrame(dtype=float)]
)
@pytest.mark.parametrize(
    "bfunc",
    [
        unsupported_return_type_str,
        unsupported_return_type_bool,
        unsupported_return_type_list,
        unsupported_return_type_tuple,
        unsupported_return_type_set,
    ],
)
def test_no_signature(parallel, data_input, bfunc) -> None:
    with pytest.raises(UnsupportedReturnType):
        bootstrap(data_input, bfunc, parellel=parallel)
