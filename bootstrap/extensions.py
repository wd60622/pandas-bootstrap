"""Extensions for pandas DataFrames and Series. Access with the `boot` attribute of a DataFrame or Series."""
import pandas as pd

from bootstrap.bootstrap import bootstrap, BFUNC


class AccessorMixin:
    def __init__(self, obj):
        self._obj = obj

    def get_samples(
        self, bfunc: BFUNC, B: int = 100, sample_kwargs=None, **kwargs
    ) -> pd.DataFrame:
        return bootstrap(
            self._obj, bfunc=bfunc, B=B, sample_kwargs=sample_kwargs, **kwargs
        )


@pd.api.extensions.register_dataframe_accessor("boot")
class DataFrameBootstrapAccessor(AccessorMixin):
    """Bootstrap accessor for pandas DataFrames.

    Example:
        Bootstrap the mean of each column in a DataFrame.

        ```python
        import pandas as pd
        import bootstrap

        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [6, 7, 8, 9, 10],
        })

        def mean_of_columns(df):
            return df.mean(numeric_only=True)

        df_bootstrap: pd.DataFrame = df.boot.get_samples(bfunc=mean_of_columns, B=100)
        ```

    """


@pd.api.extensions.register_series_accessor("boot")
class SeriesBootstrapAccessor(AccessorMixin):
    """Bootstrap accessor for pandas Series.

    Example:
        Bootstrap the mean of a Series.

        ```python
        import pandas as pd
        import bootstrap

        ser = pd.Series([1, 2, 3, 4, 5])

        def mean_of_series(ser):
            return ser.mean()

        ser_bootstrap: pd.Series = ser.boot.get_samples(bfunc=mean_of_series, B=100)
        ```

    """
