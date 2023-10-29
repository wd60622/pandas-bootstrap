from inspect import signature, Signature
from typing import Any, Callable, Dict, Union, Optional, Tuple

import numpy as np
import pandas as pd


BFUNC_INPUT = Union[pd.DataFrame, pd.Series]
BFUNC_OUTPUT = Union[pd.DataFrame, pd.Series]
BFUNC = Callable[[BFUNC_INPUT], BFUNC_INPUT]


def get_return_type(bfunc: BFUNC) -> type:
    """Get the return type of a bootstrap function."""
    sig = signature(bfunc)
    return_type = sig.return_annotation

    return return_type


class UnsupportedReturnType(Exception):
    """Raised when a bootstrap function returns an unsupported type."""


def get_return_type(bfunc: BFUNC) -> type:
    """Get the return type of a bootstrap function."""
    sig = signature(bfunc)
    return sig.return_annotation


def infer_return_type(df: BFUNC_INPUT, bfunc: BFUNC, **kwargs) -> Tuple[BFUNC, type]:
    """Infer the return type of a bootstrap function.

    Args:
        df: The DataFrame to bootstrap.
        bfunc: The bootstrap function.
        **kwargs: Keyword arguments to pass to the bootstrap function.

    Returns:
        A tuple of the bootstrap function and its return type.

    Raises:
        UnsupportedReturnType: If the bootstrap function returns an unsupported type.

    """
    df_result = bfunc(df, **kwargs)

    return_type = type(df_result)

    not_needed_to_infer_types = {pd.DataFrame, pd.Series}
    supported_numpy_types = {np.float32, np.float64, np.int32, np.int64}
    need_to_infer_types = {float, int}.union(supported_numpy_types)

    all_supported_types = not_needed_to_infer_types.union(need_to_infer_types)
    if return_type not in all_supported_types:
        raise UnsupportedReturnType(
            f"Bootstrap function must return a DataFrame, Series, float, or int, not {return_type}"
        )

    if return_type in need_to_infer_types:

        def wrapped_bfunc(df: BFUNC_INPUT, **kwargs) -> pd.Series:
            return pd.Series({"stat": bfunc(df, **kwargs)})

        return_type = pd.Series
        return wrapped_bfunc, return_type

    return bfunc, return_type


class DataFrameFunction:
    """Process a bootstrap function that returns a DataFrame."""

    append_axis: int = 0

    @staticmethod
    def name(boot_sample, i: int):
        boot_sample["sample"] = i
        return boot_sample

    @staticmethod
    def post_process(df: pd.DataFrame) -> pd.DataFrame:
        return df.set_index("sample", append=True).sort_index()


class SeriesFunction:
    """Process a bootstrap function that returns a Series."""

    append_axis: int = 1

    @staticmethod
    def name(boot_sample: pd.Series, i: int):
        boot_sample.name = i
        return boot_sample

    @staticmethod
    def post_process(s: pd.DataFrame) -> pd.DataFrame:
        s = s.T
        s.index.name = "sample"
        return s


def get_bfunc_processor(return_type: type) -> Union[DataFrameFunction, SeriesFunction]:
    """Get the appropriate bootstrap function processor."""
    if return_type == pd.DataFrame:
        return DataFrameFunction()
    elif return_type == pd.Series:
        return SeriesFunction()
    else:
        raise UnsupportedReturnType(
            f"Bootstrap function must return a DataFrame or Series, not {return_type}. Consider wrapping return in pd.Series"
        )


def bootstrap(
    bfunc_input: BFUNC_INPUT,
    bfunc: BFUNC,
    B: int = 100,
    sample_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[pd.DataFrame, pd.Series]:
    """Core bootstrap function.

    Args:
        bfunc_input: Input to bootstrap function.
        bfunc: Bootstrap function.
        B: Number of bootstrap samples.
        sample_kwargs: Keyword arguments to pass to the sampling function.
        **kwargs: Keyword arguments to pass to the bootstrap function.

    Returns:
        Bootstrap samples in a DataFrame or Series.

    Raises:
        ValueError: If sample_kwargs contains 'replace' or 'frac' keys.
        UnsupportedReturnType: If the bootstrap function returns an unsupported type.

    """
    if sample_kwargs is None:
        sample_kwargs = {}

    if "replace" in sample_kwargs or "frac" in sample_kwargs:
        raise ValueError("sample_kwargs cannot contain 'replace' or 'frac' keys.")

    if "random_state" in sample_kwargs and isinstance(
        sample_kwargs["random_state"], int
    ):
        sample_kwargs["random_state"] = np.random.default_rng(
            sample_kwargs["random_state"]
        )

    sample_kwargs["replace"] = True
    sample_kwargs["frac"] = 1

    return_type = get_return_type(bfunc)
    if return_type in {Signature.empty, float, int}:
        bfunc, return_type = infer_return_type(bfunc_input, bfunc, **kwargs)

    bfunc_processor = get_bfunc_processor(return_type)

    samples = []
    for i in range(B):
        boot_sample = bfunc(bfunc_input.sample(**sample_kwargs), **kwargs)
        boot_sample = bfunc_processor.name(boot_sample, i)
        samples.append(boot_sample)

    return (
        pd.concat(samples, axis=bfunc_processor.append_axis)
        .pipe(bfunc_processor.post_process)
        .squeeze()
    )
