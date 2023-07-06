import polars as pl
from typing import Tuple, Optional
from .type_alias import PolarsFrame
from polars.type_aliases import UniqueKeepStrategy
import polars.selectors as cs
import numpy as np
import logging

logger = logging.getLogger(__name__)

def lazy_sample(df:pl.LazyFrame, sample_frac:float, seed:int=42) -> pl.LazyFrame:
    '''
    Random sample on a lazy dataframe.
    
    Parameters
    ----------
    df
        A lazy dataframe
    sample_frac
        A number > 0 and < 1
    seed
        The random seed

    Returns
    -------
        A lazy dataframe containing the sampling query
    '''
    if sample_frac <= 0 or sample_frac >= 1:
        raise ValueError("Sample fraction must be > 0 and < 1.")

    return df.with_columns(pl.all().shuffle(seed=seed)).with_row_count()\
        .filter(pl.col("row_nr") < pl.col("row_nr").max() * sample_frac)\
        .select(df.columns)

def deduplicate(
    df: PolarsFrame
    , by: list[str]
    , keep: UniqueKeepStrategy = "first"
) -> PolarsFrame:
    '''
    A wrapper function for Polar's unique method.

    Parameters
    ----------
    df
        Either an eager or lazy dataframe
    by
        The list of columns to dedplicate by
    keep
        One of 'first', 'last', 'any', 'none'

    Returns
    -------
        A deduplicated eager/lazy frame.
    '''
    return df.unique(subset=by, keep = keep)

def simple_upsample(
    df: PolarsFrame
    , subgroup: dict[str, list] | pl.Expr
    , count:int
    , epsilon: float = 1e-2
    , include: Optional[list[str]] = None
    , exclude: Optional[list[str]] = None
    , seed: int = 42
) -> PolarsFrame:
    '''
    For records in the subgroup, we (1) sample with replacement for `count` many records
    and (2) add a small random number uniformly distributed in (-epsilon, epsilon) to all 
    the float-valued columns except those in exclude.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    subgroup
        Either a dict that looks like {"c1":[v1, v2], "c2":[v3]}, which will translates to
        pl.col("c1").is_in([v1,v2]) & pl.col("c2").is_in([v3]), or a Polars expression for 
        more complicated subgroups.
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
    seed
        The random seed

    Returns
    -------
        a lazy/eager dataframe with the subgroup amplified.

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
        new_c = pl.Series(
            c, sub[c].to_numpy() + (rng.random(size=(len(sub),)) * 2 * epsilon - epsilon)
        ).fill_nan(None) # NaN occurs when we add null with a number. Leave it null.
        sub = sub.replace_at_idx(sub.find_idx_by_name(c), new_c)

    if isinstance(df, pl.LazyFrame):
        return pl.concat([df, sub.lazy()])
    return pl.concat([df, sub])

def stratified_downsample(
    df: PolarsFrame
    , by:list[str]
    , keep:int | float
    , min_keep:int = 1
) -> PolarsFrame:
    '''
    Stratified downsampling.

    Parameters
    ----------
    df
        Either an eager or lazy dataframe
    by
        Column groups you want to use to stratify the data
    keep
        If int, keep this number of records from this subpopulation; if float, then
        keep this % of the subpopulation.
    min_keep
        Always an int. E.g. say the subpopulation only has 2 records. You set 
        keep = 0.3, then we are keeping 0.6 records, which means we are removing the entire
        subpopulation. Setting min_keep will make sure we keep at least this many of each 
        subpopulation provided that it has this many records.

    Returns
    -------
        the downsampled eager/lazy frame
    '''
    if isinstance(keep, int):
        if keep <= 0:
            raise ValueError("The argument `keep` must be a positive integer.")
        rhs = pl.lit(keep, dtype=pl.UInt64)
    elif isinstance(keep, float):
        if keep < 0. or keep >= 1.:
            raise ValueError("The argument `keep` must be >0 and <1.")
        rhs = pl.max(pl.count().over(by)*keep, min_keep)
    else:
        raise TypeError("The argument `keep` must either be a Python int or float.")

    return df.filter(
        pl.arange(0, pl.count(), dtype=pl.UInt64).shuffle().over(by) < rhs
    )

def train_test_split(
    df: PolarsFrame
    , train_frac:float = 0.75
    , seed:int = 42
) -> Tuple[PolarsFrame, PolarsFrame]:
    """
    Split polars dataframe into train and test set. If input is eager, output will be eager. If input is lazy, out
    output will be lazy. Unlike scikit-learn, this only creates the train and test dataframe. This
    will not break the dataframe into X and y and so target is not a necessary input.

    Parameters
    ----------
        df
            Either a lazy or eager dataframe to split
        train_frac
            Fraction that goes to train. Defaults to 0.75.
        seed
            the random seed.

    Returns
    -------
        the lazy or eager train and test dataframes
    """
    keep = df.columns # with_row_count will add a row_nr column. Don't need it.
    if isinstance(df, pl.DataFrame):
        # Eager group by is iterable
        p1, p2 = df.with_columns(pl.all().shuffle(seed=seed))\
                    .with_row_count().groupby(
                        pl.col("row_nr") >= len(df) * train_frac
                    )
        
        # I am not sure if False group is always returned first...
        # p1 is a 2-tuple of (True/False, the corresponding group)
        if p2[0]: # if p2[0] == True, then p1[1] is train, p2[1] is test
            return p1[1].select(keep), p2[1].select(keep) # Make sure train comes first
        return p2[1].select(keep), p1[1].select(keep)
    else: # Lazy case.
        df = df.lazy().with_columns(pl.all().shuffle(seed=seed)).with_row_count()
        df_train = df.filter(pl.col("row_nr") < pl.col("row_nr").max() * train_frac)
        df_test = df.filter(pl.col("row_nr") >= pl.col("row_nr").max() * train_frac)
        return df_train.select(keep), df_test.select(keep)