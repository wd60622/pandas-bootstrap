"""Core bootstrapping logic which powers the `boot` extension including parallelization and return type inference.

"""
from copy import copy
from inspect import signature, Signature
from typing import Any, Callable, List, Dict, Union, Optional, Tuple

import sys

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec

from joblib import Parallel, delayed

import numpy as np
import pandas as pd


BFUNC_INPUT = Union[pd.DataFrame, pd.Series]
BFUNC_OUTPUT = Union[pd.DataFrame, pd.Series]
BFUNC = Callable[[BFUNC_INPUT], BFUNC_INPUT]

P = ParamSpec("P")


class UnsupportedReturnType(Exception):
    """Raised when a bootstrap function returns an unsupported type."""


def get_return_type(bfunc: BFUNC) -> type:
    """Get the return type of a bootstrap function."""
    sig = signature(bfunc)
    return sig.return_annotation


def infer_return_type(
    df: BFUNC_INPUT, bfunc: BFUNC, **kwargs: P.kwargs
) -> Tuple[BFUNC, type]:
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


def create_inner_loop_func(
    bfunc, bfunc_input, bfunc_processor, sample_kwargs, **kwargs: P.kwargs
):
    def inner_loop(i):
        boot_sample = bfunc(bfunc_input.sample(**sample_kwargs), **kwargs)
        boot_sample = bfunc_processor.name(boot_sample, i)

        return boot_sample

    return inner_loop


def inner_loop(
    bfunc, bfunc_input, bfunc_processor, i, sample_kwargs, **kwargs: P.kwargs
):
    boot_sample = bfunc(bfunc_input.sample(**sample_kwargs), **kwargs)
    boot_sample = bfunc_processor.name(boot_sample, i)

    return boot_sample


LOOP_FUNC = Callable[[Callable[[int], BFUNC_OUTPUT], int], List[BFUNC_OUTPUT]]


def loop(inner_loop_func, B):
    samples = []
    for i in range(B):
        boot_sample = inner_loop_func(i)
        samples.append(boot_sample)

    return samples


def create_parallel_loop(parallel):
    def parallelized_loop(inner_loop_func, B):
        return parallel(delayed(inner_loop_func)(i) for i in range(B))

    return parallelized_loop


def bootstrap(
    bfunc_input: BFUNC_INPUT,
    bfunc: BFUNC,
    B: int = 100,
    sample_kwargs: Optional[Dict[str, Any]] = None,
    parallel: Optional[Parallel] = None,
    **kwargs: P.kwargs,
) -> Union[pd.DataFrame, pd.Series]:
    """Core bootstrap function.

    Args:
        bfunc_input: Input to bootstrap function.
        bfunc: Bootstrap function.
        B: Number of bootstrap samples.
        sample_kwargs: Keyword arguments to pass to the sampling function.
        parallel: Parallelization object.
        **kwargs: Keyword arguments to pass to the bootstrap function.

    Returns:
        Bootstrap samples in a DataFrame or Series.

    Raises:
        ValueError: If sample_kwargs contains 'replace' or 'frac' keys.
        UnsupportedReturnType: If the bootstrap function returns an unsupported type.

    """
    sample_kwargs = {} if sample_kwargs is None else copy(sample_kwargs)

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

    loop_func = loop if parallel is None else create_parallel_loop(parallel)
    inner_loop_func = create_inner_loop_func(
        bfunc, bfunc_input, bfunc_processor, sample_kwargs, **kwargs
    )
    samples = loop_func(inner_loop_func=inner_loop_func, B=B)

    return (
        pd.concat(samples, axis=bfunc_processor.append_axis)
        .pipe(bfunc_processor.post_process)
        .squeeze()
    )
