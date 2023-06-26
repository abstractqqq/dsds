import polars as pl
from typing import Tuple
from .type_alias import (
    PolarsFrame
)

# Split? Or create an indicator column?
def recent_split(df:pl.DataFrame, sort_col:str, keep:int, keep_pct:float=-1.) -> pl.DataFrame:
    pass


def train_test_split_lazy(
    df: PolarsFrame
    , train_fraction: float = 0.75
    , seed:int = 42
) -> Tuple[PolarsFrame, PolarsFrame]:
    """Split polars dataframe into two sets. If input is eager, output will be eager. If input is lazy, out
    output will be lazy. This is a copy and paste from Laurent's answer on stackoverflow. See
    source.

    Args:
        df (pl.DataFrame): Dataframe to split
        train_fraction (float, optional): Fraction that goes to train. Defaults to 0.75.
    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Tuple of train and test dataframes

    Source:
        https://stackoverflow.com/questions/76499865/splitting-a-lazyframe-into-two-frames-by-fraction-of-rows-to-make-a-train-test-s
    """
    

    df = df.lazy().with_columns(pl.all().shuffle(seed=seed)).with_row_count()\
        
        
    df_train = df.filter(pl.col("row_nr") < pl.col("row_nr").max() * train_fraction)
    df_test = df.filter(pl.col("row_nr") >= pl.col("row_nr").max() * train_fraction)
    return df_train, df_test