import polars as pl
from typing import Tuple

# Split? Or create an indicator column?
def recent_split(df:pl.DataFrame, sort_col:str, keep:int, keep_pct:float=-1.) -> pl.DataFrame:
    pass


def train_test_split(df:pl.DataFrame, train_pct:float, random_state:int=42) -> Tuple[pl.DataFrame, pl.DataFrame]:
    pass