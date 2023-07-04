import polars as pl
from .type_alias import PolarsFrame

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

def stratified_downsample(
    df: PolarsFrame
    , groupby:list[str]
    , keep:int | float
    , min_keep:int = 1
) -> PolarsFrame:
    '''Stratified downsampling.

        Arguments:
            df: either an eager or lazy dataframe
            groupby: groups you want to use to stratify the data
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
    else:
        if keep < 0 or keep >= 1:
            raise ValueError("The argument `keep` must be >0 and <1.")
        rhs = pl.max(pl.count().over(groupby)*keep, min_keep)

    return df.filter(
        pl.arange(0, pl.count(), dtype=pl.UInt64).shuffle().over(groupby) < rhs
    )