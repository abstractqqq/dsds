import pytest
import polars as pl
import pandas as pd
import dsds.sample as sa
from dsds.type_alias import PolarsFrame
from sklearn.model_selection import train_test_split 

def dsds_train_test_split(df: PolarsFrame, train_frac:float = 0.75) -> tuple[pl.DataFrame, pl.DataFrame]:
    return sa.train_test_split(df, train_frac=train_frac)

def sklearn_train_test_split(df: pd.DataFrame, train_frac:float = 0.75) -> list:
    X_train, X_test = train_test_split(df, train_size=train_frac)
    return X_train, X_test

@pytest.mark.benchmark
def test_split_on_2mm_dsds(benchmark):
    df = pl.read_parquet("./data/dunnhumby.parquet")
    _ = benchmark(
        dsds_train_test_split, df, train_frac=0.75
    )
    return 123

@pytest.mark.benchmark
def test_split_on_2mm_sklearn(benchmark):
    df = pd.read_parquet("./data/dunnhumby.parquet")
    _ = benchmark(
        sklearn_train_test_split, df, train_frac=0.75
    )
    return 123