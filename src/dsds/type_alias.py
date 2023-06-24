from typing import TypeAlias
import polars as pl 

PolarsFrame:TypeAlias = pl.DataFrame | pl.LazyFrame