"""Extensions for pandas DataFrames and Series. Access with the `boot` attribute of a DataFrame or Series.

Available after importing the package:

```python
import bootstrap

df = pd.DataFrame(...)
df.boot.get_samples(...)

ser = pd.Series(...)
ser.boot.get_samples(...)
```

"""
import sys
from typing import Any, Dict, Optional, Union

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec

from joblib import Parallel

import pandas as pd

from bootstrap.bootstrap import bootstrap, BFUNC


P = ParamSpec("P")


class AccessorMixin:
    """Common functionality for DataFrame and Series accessors."""

    def __init__(self, obj):
        self._obj = obj

    def get_samples(
        self,
        bfunc: BFUNC,
        B: int = 100,
        sample_kwargs: Dict[str, Any] = None,
        parallel: Optional[Parallel] = None,
        **kwargs: P.kwargs,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Get bootstrap samples of the object.

        Args:
            bfunc: Bootstrap function.
            B: Number of bootstrap samples.
            sample_kwargs: Keyword arguments to pass to the sample method of the object.
            parallel: Joblib Parallel object.
            kwargs: Keyword arguments to pass to the bootstrap function.

        Returns:
            Bootstrap samples in a Series or DataFrame. Depends on the return type of the bootstrap function.

        """
        return bootstrap(
            self._obj,
            bfunc=bfunc,
            B=B,
            sample_kwargs=sample_kwargs,
            parallel=parallel,
            **kwargs,
        )


@pd.api.extensions.register_dataframe_accessor("boot")
class DataFrameBootstrapAccessor(AccessorMixin):
    """Bootstrap accessor for pandas DataFrames.

    Examples:
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

    Examples:
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
