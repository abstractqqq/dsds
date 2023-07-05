import polars as pl
from typing import Tuple
from .type_alias import PolarsFrame
from polars.type_aliases import UniqueKeepStrategy

def lazy_sample(df:pl.LazyFrame, sample_frac:float, seed:int=42) -> pl.LazyFrame:
    '''Random sample on a lazy dataframe.
    
        Arguments:
            df: a lazy dataframe
            sample_frac: a number > 0 and < 1
            seed: random seed

        Returns:
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
    '''A wrapper function for Polar's unique method. 
        Arguments:
            df: either an eager or lazy dataframe
            by: the list of columns to dedplicate by
            keep: one of 'first', 'last', 'any', 'none'

        Returns:
            A deduplicated eager/lazy frame.
    '''
    return df.unique(subset=by, keep = keep)

def stratified_downsample(
    df: PolarsFrame
    , by:list[str]
    , keep:int | float
    , min_keep:int = 1
) -> PolarsFrame:
    '''Stratified downsampling.

        Arguments:
            df: either an eager or lazy dataframe
            by: column group you want to use to stratify the data
            keep: if int, keep this number of records from this subpopulation; if float, then
            keep this % of the subpopulation.
            min_keep: always an int. E.g. say the subpopulation only has 2 records. You set 
            keep = 0.3, then we are keeping 0.6 records, which means we are removing the entire
            subpopulation. Setting min_keep will make sure we keep at least this many of each 
            subpopulation provided that it has this many records.

        Returns:
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
    , train_frac: float = 0.75
    , seed:int = 42
) -> Tuple[PolarsFrame, PolarsFrame]:
    """Split polars dataframe into train and test set. If input is eager, output will be eager. If input is lazy, out
    output will be lazy.

    Arguments:
        df: Dataframe to split
        train_frac: Fraction that goes to train. Defaults to 0.75.
        seed: the random seed.
    Returns:
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