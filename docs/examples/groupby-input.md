---
comments: true
---
# GroupBy Input

The `DataFrameGroupBy` and `SeriesGroupBy` do not have `boot` attributes.
However, the `groupby` method can be used in the `bfunc`.

!!! note
    Each group will have different sample sizes than the original dataset.

Check out the example [here](../datasets.md#bootstrap.datasets.different_mean_and_sigma) which uses a `bfunc` with the `groupby` method.
