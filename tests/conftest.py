import pytest

import numpy as np
import pandas as pd


@pytest.fixture
def df() -> pd.DataFrame:
    n_samples = 100
    data = np.random.normal(size=(n_samples, 2))
    frame = pd.DataFrame(data, columns=["x", "y"])
    frame["group"] = np.random.choice(["A", "B"], size=n_samples)

    return frame


@pytest.fixture
def series() -> pd.Series:
    n_samples = 100
    data = np.random.normal(size=n_samples)
    series = pd.Series(data, name="x")

    return series
