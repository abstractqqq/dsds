from .type_alias import (
    PolarsFrame
)
import polars as pl
from polars import LazyFrame
from typing import Concatenate, ParamSpec, Callable

P = ParamSpec("P")

class PipeBuilder2():
    
    def __init__(self, df:PolarsFrame):
        self.data:LazyFrame = df.lazy()
        self.dummy = pl.LazyFrame()

    def pipe(self, function: Callable[Concatenate[PolarsFrame, P], PolarsFrame], *args: P.args, **kwargs: P.kwargs):
        self.data = self.data.pipe(function=function, *args, **kwargs)

        return self 
        