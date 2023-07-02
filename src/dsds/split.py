import polars as pl
from typing import Tuple
from .type_alias import (
    PolarsFrame
)

# Split? Or create an indicator column?
# def recent_split(df:pl.DataFrame, sort_col:str, keep:int, keep_pct:float=-1.) -> pl.DataFrame:
#     pass


def train_test_split(
    df: PolarsFrame
    , train_fraction: float = 0.75
    , seed:int = 42
) -> Tuple[PolarsFrame, PolarsFrame]:
    """Split polars dataframe into train and test set. If input is eager, output will be eager. If input is lazy, out
    output will be lazy.

    Arguments:
        df (pl.DataFrame): Dataframe to split
        train_fraction (float): Fraction that goes to train. Defaults to 0.75.
        seed (int): the random seed.
    Returns:
        Tuple[PolarsFrame, PolarsFrame]: in the train then test order.

    Source:
        https://stackoverflow.com/questions/76499865/splitting-a-lazyframe-into-two-frames-by-fraction-of-rows-to-make-a-train-test-s
    """
    keep = df.columns # with_row_count will add a row_nr column. Don't need it.
    if isinstance(df, pl.DataFrame):
        # Eager group by is iterable
        p1, p2 = df.with_columns(pl.all().shuffle(seed=seed))\
                    .with_row_count().groupby(
                        pl.col("row_nr") >= len(df) * train_fraction
                    )
        
        # I am not sure if False group is always returned first...
        # p1 is a 2-tuple of (True/False, the corresponding group)
        if p2[0]: # if p2[0] == True, then p1[1] is train, p2[1] is test
            return p1[1].select(keep), p2[1].select(keep) # Make sure train comes first
        return p2[1].select(keep), p1[1].select(keep)
    else: # Lazy case.
        df = df.lazy().with_columns(pl.all().shuffle(seed=seed)).with_row_count()
        df_train = df.filter(pl.col("row_nr") < pl.col("row_nr").max() * train_fraction)
        df_test = df.filter(pl.col("row_nr") >= pl.col("row_nr").max() * train_fraction)
        return df_train.select(keep), df_test.select(keep)