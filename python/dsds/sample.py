from typing import (
    Tuple
    , Optional
    , Union
)
from collections.abc import Iterator
from .type_alias import PolarsFrame
from polars.type_aliases import UniqueKeepStrategy
from itertools import product
import polars as pl
import polars.selectors as cs
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

def lazy_sample(
    df:pl.LazyFrame
    , sample_frac: Optional[float] = 0.75
    , sample_amt: Optional[int] = None
    , seed:int=42
    , persist:bool=False
) -> pl.LazyFrame:
    '''
    Random sample on a lazy dataframe.

    Set persist = True if this needs to be remembered by the blueprint.
    
    Parameters
    ----------
    df
        A lazy dataframe
    sample_frac
        A number > 0 and < 1. If this and sample_amt are both not net, this defaults to 0.75
    sample_amt
        If this is set, sample this amount, and sample_frac will be ignored
    seed
        The random seed
    persist
        If true, this step will be kept in the blueprint
    '''
    if sample_amt is None:
        if sample_frac is None:
            frac = 0.75
        elif sample_frac <= 0 or sample_frac >= 1:
            raise ValueError("Sample fraction must be > 0 and < 1.")
        else:
            frac = sample_frac

        output = df.with_columns(pl.all().shuffle(seed=seed)).with_row_count()\
            .set_sorted("row_nr")\
            .filter(pl.col("row_nr") < pl.col("row_nr").max() * frac)\
            .select(df.columns)

    else:
        output = df.with_columns(pl.all().shuffle(seed=seed)).with_row_count()\
            .set_sorted("row_nr")\
            .filter(pl.col("row_nr") < sample_amt)\
            .select(df.columns)

    if persist:
        output = output.blueprint.add_func(df, lazy_sample, kwargs = {"sample_frac":sample_frac
                                                                    , "sample_amt":sample_amt
                                                                    , "seed":seed})
    return output

def deduplicate(
    df: PolarsFrame
    , subset: list[str]
    , keep: UniqueKeepStrategy = "any"
    , persist: bool = False
) -> PolarsFrame:
    '''
    A wrapper function for Polar's unique method. It deduplicates the dataframe by the columns in `by`.

    Set persist = True if this needs to be remembered by the blueprint.

    Parameters
    ----------
    df
        Either an eager or lazy dataframe
    by
        The list of columns to deduplicate by
    keep
        One of 'first', 'last', 'any', 'none'. Default is 'any', which is somewhat random.
    persist
        If true, this step will be kept in the blueprint
    '''
    output = df.unique(subset=subset, keep = keep)
    if isinstance(df, pl.LazyFrame) and persist:
        return output.blueprint.add_func(df, deduplicate, kwargs = {"subset":subset,"keep":keep})
    return output

def simple_upsample(
    df: PolarsFrame
    , subgroup: Union[dict[str, list], pl.Expr]
    , count:int
    , epsilon: float = 1e-2
    , include: Optional[list[str]] = None
    , exclude: Optional[list[str]] = None
    , positive: bool = False
    , seed: int = 42
    , persist: bool = False
) -> PolarsFrame:
    '''
    For records in the subgroup, we (1) sample with replacement for `count` many records
    and (2) add a small random number uniformly distributed in (-epsilon, epsilon) to all 
    the float-valued columns except those in exclude.

    Set persist = True if this needs to be remembered by the blueprint.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    subgroup
        Either a dict that looks like {"c1":[v1, v2], "c2":[v3]}, which will translates to
        pl.col("c1").is_in([v1,v2]) & pl.col("c2").is_in([v3]), or a Polars expression for 
        more complicated subgroup. Only records in the subgroup will be upsampled
    count
        How many more to add
    epsilon
        The random noise to be added will be uniformly distributed with bounds (-epsilon, epsilon)
    include
        Columns to which we may add some small random noise. If provided, a random noise will be 
        added to only the columns. If not provided, all float-valued columns will be used. 
        If no float-valued columns exist, then no noise will be added.
    exclude
        Columns to which random noises should not be added
    positive
        If true, then the interval for the random noise will be (0, epsilon)
    seed
        The random seed
    persist
        If true, this step will be kept in the blueprint

    Examples
    --------
    >>> df.groupby("One_Hot_Test").count()
    shape: (3, 2)
    ┌──────────────┬───────┐
    │ One_Hot_Test ┆ count │
    │ ---          ┆ ---   │
    │ str          ┆ u32   │
    ╞══════════════╪═══════╡
    │ B            ┆ 114   │
    │ C            ┆ 103   │
    │ A            ┆ 783   │
    └──────────────┴───────┘
    >>> upsampled = simple_upsample(df, subgroup={"One_Hot_Test":["B", "C"]}, count=200, exclude=["Clicked on Ad"])
    >>> upsampled.groupby("One_Hot_Test").count()
    shape: (3, 2)
    ┌──────────────┬───────┐
    │ One_Hot_Test ┆ count │
    │ ---          ┆ ---   │
    │ str          ┆ u32   │
    ╞══════════════╪═══════╡
    │ C            ┆ 186   │
    │ A            ┆ 783   │
    │ B            ┆ 231   │
    └──────────────┴───────┘
    '''
    if include is None:
        if exclude is None:
            to_add_noise = df.select(cs.by_dtype(pl.Float32, pl.Float64)).columns
        else:
            to_add_noise = df.select(cs.by_dtype(pl.Float32, pl.Float64)& ~cs.by_name(exclude)).columns
    else:
        if exclude is None:
            to_add_noise = include
        else:
            to_add_noise = (f for f in include if f not in exclude)

    # Should be small, because this is the whole point of upsampling
    if isinstance(subgroup, pl.Expr):
        sub = (
            df.lazy().filter(subgroup)
            .collect()
            .sample(n=count, with_replacement=True)
        )
    elif isinstance(subgroup, dict):
        filter_expr = pl.lit(True)
        for c, vals in subgroup.items():
            if isinstance(vals, list):
                filter_expr = filter_expr & pl.col(c).is_in(vals)
            else:
                logger.warn(f"The value for key `{c}` is not a list. Skipped.")
        sub = (
            df.lazy().filter(filter_expr)
            .collect()
            .sample(n=count, with_replacement=True)
        )
    else:
        raise TypeError("The `subgroup` argument must be either a Polars Expr or a dict[str, list]")

    rng = np.random.default_rng(seed)
    for c in to_add_noise:
        if positive:
            noise = rng.random(size=(len(sub),)) * epsilon
        else:
            noise = rng.random(size=(len(sub),)) * 2 * epsilon - epsilon
        # NaN occurs in the to_numpy() step. So the series might contain NaN. Replace them with Null.
        new_c = pl.Series(c, sub[c].to_numpy() + noise).fill_nan(None) 
        sub = sub.replace_at_idx(sub.find_idx_by_name(c), new_c)

    if isinstance(df, pl.LazyFrame):
        output = pl.concat([df, sub.lazy()])
        if persist:
            output = output.blueprint.add_func(df, simple_upsample, kwargs = {"subgroup":subgroup,"count":count
                                                                            ,"epsilon":epsilon,"include":include
                                                                            ,"exclude":exclude,"positive":positive
                                                                            ,"seed":seed})
        return output
    return pl.concat([df, sub])

def simple_downsample(
    df: PolarsFrame
    , subgroup: Union[dict[str, list], pl.Expr]
    , sample_frac: float
    , persist: bool = False
) -> PolarsFrame:
    '''
    Downsample by the given fraction on the subgroup.

    Set persist = True if this needs to be remembered by the blueprint.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    subgroup
        Either a dict that looks like {"c1":[v1, v2], "c2":[v3]}, which will translates to
        pl.col("c1").is_in([v1,v2]) & pl.col("c2").is_in([v3]), or a Polars expression for 
        more complicated subgroup. Only records in the subgroup will be downsampled
    sample_frac
        The percentage to downsample population in the subgroup to
    persist
        If true, this step will be kept in the blueprint
    '''
    if isinstance(subgroup, pl.Expr):
        expr = subgroup
    elif isinstance(subgroup, dict):
        expr = pl.lit(True)
        for c, vals in subgroup.items():
            if isinstance(vals, list):
                expr = expr & pl.col(c).is_in(vals)
            else:
                logger.warn(f"The value for key `{c}` is not a list. Skipped.")
    else:
        raise TypeError("The `subgroup` argument must be either a Polars Expr or a dict[str, list]")

    # lazy if df lazy, eager if df eager
    output = df.with_columns(
        pl.when(expr).then(True).otherwise(False).alias("~!")
    ).filter(
        (~pl.col("~!"))
        | (pl.int_range(0, pl.count()).over("~!").shuffle() < pl.count().over("~!").max() * sample_frac)
    ).select(df.columns)

    if isinstance(df, pl.LazyFrame):
        if persist:
            output = output.blueprint.add_func(df, simple_downsample
                                                , kwargs={"subgroup":subgroup,"sample_frac":sample_frac})
        return output
    return output

def stratified_downsample(
    df: PolarsFrame
    , group:list[str]
    , keep: Union[int, float]
    , min_keep:int = 1
    , persist: bool = False
) -> PolarsFrame:
    '''
    Stratified downsampling.

    Set persist = True if this needs to be remembered by the blueprint.

    Parameters
    ----------
    df
        Either an eager or lazy dataframe
    group
        Column group you want to use to stratify the data
    keep
        If int, keep this number of records from this subpopulation; if float, then
        keep this % of the subpopulation.
    min_keep
        Always an int. E.g. say the subpopulation only has 2 records. You set 
        keep = 0.3, then we are keeping 0.6 records, which means we are removing the entire
        subpopulation. Setting min_keep will make sure we keep at least this many of each 
        subpopulation provided that it has this many records.
    persist
        If true, this step will be kept in the blueprint
    '''
    if isinstance(keep, int):
        if keep <= 0:
            raise ValueError("The argument `keep` must be a positive integer.")
        rhs = pl.lit(keep, dtype=pl.UInt64)
    elif isinstance(keep, float):
        if keep < 0. or keep >= 1.:
            raise ValueError("The argument `keep` must be >0 and <1.")
        rhs = pl.max(pl.count().over(group)*keep, min_keep)
    else:
        raise TypeError("The argument `keep` must either be a Python int or float.")

    output = df.filter(
        pl.int_range(0, pl.count(), dtype=pl.UInt64).shuffle().over(group) < rhs
    )
    if isinstance(df, pl.LazyFrame) and persist:
        return output.blueprint.add_func(df, stratified_downsample
                                        , kwargs={"group":group, "keep":keep, "min_keep":min_keep})
    return output

def train_test_split(
    df: PolarsFrame
    , train_frac:float = 0.75
    , seed:int = 42
) -> Tuple[PolarsFrame, PolarsFrame]:
    """
    Split polars dataframe into train and test set. If input is eager, output will be eager. If input is lazy, out
    output will be lazy. Unlike scikit-learn, this only creates the train and test dataframe. This will not break 
    the dataframe into X and y and so target is not a necessary input.

    Parameters
    ----------
        df
            Either a lazy or eager dataframe to split
        train_frac
            Fraction that goes to train. Defaults to 0.75.
        seed
            the random seed.
    """
    keep = df.columns # with_row_count will add a row_nr column. Don't need it.
    if isinstance(df, pl.DataFrame):
        # Eager group by is iterable
        p1, p2 = df.with_columns(pl.all().shuffle(seed=seed))\
                    .with_row_count()\
                    .set_sorted("row_nr")\
                    .groupby(
                        pl.col("row_nr") >= len(df) * train_frac
                    )
        # Sometimes p1 is true, p2 is false, sometimes otherwise
        if p2[0]: # if p2[0] == True, then p1[1] is train, p2[1] is test
            return p1[1].select(keep), p2[1].select(keep) # Make sure train comes first
        return p2[1].select(keep), p1[1].select(keep)
    else: # Lazy case.
        df = df.lazy().with_columns(pl.all().shuffle(seed=seed)).with_row_count().set_sorted("row_nr")
        df_train = df.filter(pl.col("row_nr") < pl.col("row_nr").max() * train_frac)
        df_test = df.filter(pl.col("row_nr") >= pl.col("row_nr").max() * train_frac)
        return df_train.select(keep), df_test.select(keep)

# Make a monthly split for monthly progression version too.

def time_series_split(
    df: PolarsFrame
    , n_splits: int = 5
    , test_size: Optional[int] = None
    , sort_col: Optional[str] = None
    , offset: int = 0
    , gap: int = 0
) -> Iterator[Tuple[pl.DataFrame, pl.DataFrame]]:
    '''
    Creates a time series validator as an iterator of (train, test) eager frames.

    Parameters
    ----------
    df
        Either a lazy or eager dataframe to split
    n_splits
        The number of splits to make
    test_size
        If not provided, will default to len(df)//(n_splits+1). Raise error if < 1.
    sort_col
        Whether to sort df by the given column. If none, don't sort
    offset
        For example, if set to 100, then first 100 rows will always be part of train.
    gap
        Gap between train and test.

    Example
    -------
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> df = pl.from_numpy(X).insert_at_idx(2, pl.Series("y",y))
    >>> print(df)
    shape: (6, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 1        ┆ 2        ┆ 1   │
    │ 3        ┆ 4        ┆ 2   │
    │ 1        ┆ 2        ┆ 3   │
    │ 3        ┆ 4        ┆ 4   │
    │ 1        ┆ 2        ┆ 5   │
    │ 3        ┆ 4        ┆ 6   │
    └──────────┴──────────┴─────┘
    >>> for train, test in sa.time_series_split(df, n_splits=3, test_size=2): # only 2 folds will be generated
    >>>     print("train:", train)
    >>>     print("test:", test)
    WARNING:dsds.sample:Fold 0 is empty because of constraints imposed by input parameters. Skipped.
    train: shape: (2, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 1        ┆ 2        ┆ 1   │
    │ 3        ┆ 4        ┆ 2   │
    └──────────┴──────────┴─────┘
    test: shape: (2, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 1        ┆ 2        ┆ 3   │
    │ 3        ┆ 4        ┆ 4   │
    └──────────┴──────────┴─────┘
    train: shape: (4, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 1        ┆ 2        ┆ 1   │
    │ 3        ┆ 4        ┆ 2   │
    │ 1        ┆ 2        ┆ 3   │
    │ 3        ┆ 4        ┆ 4   │
    └──────────┴──────────┴─────┘
    test: shape: (2, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 1        ┆ 2        ┆ 5   │
    │ 3        ┆ 4        ┆ 6   │
    └──────────┴──────────┴─────┘
    >>> for train, test in sa.time_series_split(df, n_splits=5):
    >>>     print("train:", train)
    >>>     print("test:", test)
    train: shape: (1, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 1        ┆ 2        ┆ 1   │
    └──────────┴──────────┴─────┘
    test: shape: (1, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 3        ┆ 4        ┆ 2   │
    └──────────┴──────────┴─────┘
    train: shape: (2, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 1        ┆ 2        ┆ 1   │
    │ 3        ┆ 4        ┆ 2   │
    └──────────┴──────────┴─────┘
    test: shape: (1, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 1        ┆ 2        ┆ 3   │
    └──────────┴──────────┴─────┘
    train: shape: (3, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 1        ┆ 2        ┆ 1   │
    │ 3        ┆ 4        ┆ 2   │
    │ 1        ┆ 2        ┆ 3   │
    └──────────┴──────────┴─────┘
    test: shape: (1, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 3        ┆ 4        ┆ 4   │
    └──────────┴──────────┴─────┘
    train: shape: (4, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 1        ┆ 2        ┆ 1   │
    │ 3        ┆ 4        ┆ 2   │
    │ 1        ┆ 2        ┆ 3   │
    │ 3        ┆ 4        ┆ 4   │
    └──────────┴──────────┴─────┘
    test: shape: (1, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 1        ┆ 2        ┆ 5   │
    └──────────┴──────────┴─────┘
    train: shape: (5, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 1        ┆ 2        ┆ 1   │
    │ 3        ┆ 4        ┆ 2   │
    │ 1        ┆ 2        ┆ 3   │
    │ 3        ┆ 4        ┆ 4   │
    │ 1        ┆ 2        ┆ 5   │
    └──────────┴──────────┴─────┘
    test: shape: (1, 3)
    ┌──────────┬──────────┬─────┐
    │ column_0 ┆ column_1 ┆ y   │
    │ ---      ┆ ---      ┆ --- │
    │ i32      ┆ i32      ┆ i32 │
    ╞══════════╪══════════╪═════╡
    │ 3        ┆ 4        ┆ 6   │
    └──────────┴──────────┴─────┘
    '''
    if n_splits < 2:
        raise ValueError("Input `n_splits` must be >= 2.")
    
    keep = df.columns
    if sort_col is None:
        df_local = df.with_row_count(offset=1).set_sorted("row_nr")
    else:
        df_local = df.sort(by=sort_col).with_row_count(offset=1).set_sorted("row_nr")

    if test_size is None:
        if isinstance(df, pl.LazyFrame):
            test_size = pl.col("row_nr").max().floordiv(n_splits + 1)
        else:
            test_size = len(df)//(n_splits + 1)
    elif test_size < 1:
        raise ValueError(f"Input `test_size` must be >= 1, not {test_size}.")
    
    for i, j in enumerate(range(n_splits, 0, -1)):
        rhs = offset + pl.col("row_nr").max() - j * test_size + 1
        train = df_local.lazy().filter(pl.col("row_nr") < rhs).select(keep).collect()
        test = df_local.lazy().filter(
                pl.col("row_nr").is_between(rhs + gap, rhs + gap + test_size, closed="left")
            ).select(keep).collect()
        
        if len(train) == 0 or len(test) == 0:
            logger.warn(f"Fold {i} is empty because of constraints imposed by input parameters. Skipped.")
        else:
            yield train, test
    
def bootstrap(
    df: PolarsFrame
    , times: int
    , sample_frac: Optional[float] = 0.25
    , sample_amt: Optional[int] = None
) -> Iterator[pl.DataFrame]:
    """
    Returns an iterator (generator) where each element is a sample from the underlying df. The dataframes in
    the iterator will be eager.

    Parameters
    ----------
        df
            Either a lazy or eager dataframe to split
        times
            The number of times to sample. Total number of yields for this generator
        sample_frac
            The fraction to sample. Defaults to 0.25
        sample_amt
            If set, sample this amount instead of fraction
    """
    start = random.randint(0, 100_000)
    if isinstance(df, pl.LazyFrame):
        return (
            lazy_sample(df, sample_frac=sample_frac, sample_amt = sample_amt, seed=i).collect()
            for i in range(start, start + times)
        )
    else:
        return (
            df.sample(n = sample_amt, fraction=sample_frac, seed=i)
            for i in range(start, start + times)
        )

def col_subsample(
    df: PolarsFrame
    , times: int
    , sample_amt: int
    , always_keep: Optional[list[str]] = None
) -> Iterator[list[str]]:
    """
    Returns an iterator (generator) where each element is a subset of column names of df.

    Parameters
    ----------
        df
            Either a lazy or eager dataframe to split
        times
            The number of times to sample. Total number of yields for this generator
        sample_amt
            The number of columns to sample. Will not exceed the max number of columns
        always_keep
            Columns you want to always keep in the subsample. If always_keep = ['target'] and sample_amt = 2, then each 
            time 3 columns will be returned, 'target' and 2 other which are randomly sampled. 
    """
    if always_keep is None:
        keep = []
        to_sample = df.columns
    else:
        keep = always_keep
        to_sample = [c for c in df.columns if c not in keep]
    
    for _ in range(times):
        random.shuffle(to_sample)
        yield to_sample[:sample_amt] + keep

def segmentation(
    df: PolarsFrame
    , segments: list[list[str]]
) -> Iterator[Tuple[str, pl.DataFrame]]:
    '''
    Returns an iterator of (segment name, eager sub-dataframes).

    Parameters
    ----------
    df
        Either a lazy or an eager Polars dataframe
    segments
        A list of lists, where each element list represents the columns which you want to use as segments
    
    Example
    -------
    >>> import dsds.sample as sa
    ... import polars as pl
    ... df = pl.DataFrame({
    ...     "group":[1,1,2,2],
    ...     "state": ["TX","TX","TX","NY",],
    ...     "id": [1,2,3,4],
    ...     "value": [100,200,300,-50]
    ... })
    ... segments = [["group"],["group", "state"]]
    ... for segment, sub_df in sa.segmentation(df, segments=segments):
    ...     print(segment)
    >>>     print(sub_df)
    INFO:dsds.sample:The segment group == 1,state == NY has no record. Skipped.
    group==1 
    shape: (2, 4)
    ┌───────┬───────┬─────┬───────┐
    │ group ┆ state ┆ id  ┆ value │
    │ ---   ┆ ---   ┆ --- ┆ ---   │
    │ i64   ┆ str   ┆ i64 ┆ i64   │
    ╞═══════╪═══════╪═════╪═══════╡
    │ 1     ┆ TX    ┆ 1   ┆ 100   │
    │ 1     ┆ TX    ┆ 2   ┆ 200   │
    └───────┴───────┴─────┴───────┘
    group==2 
    shape: (2, 4)
    ┌───────┬───────┬─────┬───────┐
    │ group ┆ state ┆ id  ┆ value │
    │ ---   ┆ ---   ┆ --- ┆ ---   │
    │ i64   ┆ str   ┆ i64 ┆ i64   │
    ╞═══════╪═══════╪═════╪═══════╡
    │ 2     ┆ TX    ┆ 3   ┆ 300   │
    │ 2     ┆ NY    ┆ 4   ┆ -50   │
    └───────┴───────┴─────┴───────┘
    group==1, state==TX 
    shape: (2, 4)
    ┌───────┬───────┬─────┬───────┐
    │ group ┆ state ┆ id  ┆ value │
    │ ---   ┆ ---   ┆ --- ┆ ---   │
    │ i64   ┆ str   ┆ i64 ┆ i64   │
    ╞═══════╪═══════╪═════╪═══════╡
    │ 1     ┆ TX    ┆ 1   ┆ 100   │
    │ 1     ┆ TX    ┆ 2   ┆ 200   │
    └───────┴───────┴─────┴───────┘
    group==2, state==NY 
    shape: (1, 4)
    ┌───────┬───────┬─────┬───────┐
    │ group ┆ state ┆ id  ┆ value │
    │ ---   ┆ ---   ┆ --- ┆ ---   │
    │ i64   ┆ str   ┆ i64 ┆ i64   │
    ╞═══════╪═══════╪═════╪═══════╡
    │ 2     ┆ NY    ┆ 4   ┆ -50   │
    └───────┴───────┴─────┴───────┘
    group==2, state==TX 
    shape: (1, 4)
    ┌───────┬───────┬─────┬───────┐
    │ group ┆ state ┆ id  ┆ value │
    │ ---   ┆ ---   ┆ --- ┆ ---   │
    │ i64   ┆ str   ┆ i64 ┆ i64   │
    ╞═══════╪═══════╪═════╪═══════╡
    │ 2     ┆ TX    ┆ 3   ┆ 300   │
    └───────┴───────┴─────┴───────┘
    '''
    for seg in segments:
        temp = df.lazy().unique(subset=seg).select(seg)
        for comb in product(*temp.collect().get_columns()):
            expr = pl.lit(True)
            to_str = []
            for se, v in zip(seg, comb):
                expr = expr & pl.col(se).eq(v)
                to_str.append(f"{se}=={v}")
            output = df.lazy().filter(expr).collect()
            seg_name = ", ".join(to_str)
            if len(output) > 0:
                yield seg_name, output
            else:
                logger.info(f"The segment {seg_name} has no record. Skipped.")
    