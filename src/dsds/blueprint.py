from .type_alias import (
    PolarsFrame
)
from pathlib import Path
import polars as pl
from polars import LazyFrame
from typing import Concatenate, ParamSpec, Callable

# DO NOT USE, if you don't know what is going on...
# 
#
#

# P = ParamSpec("P")

# class PipeBuilder2():
    
#     def __init__(self, df:PolarsFrame):
#         self.data:LazyFrame = df.lazy()
#         self.dummy = pl.LazyFrame()

#     def pipe(self, function: Callable[Concatenate[PolarsFrame, P], PolarsFrame], *args: P.args, **kwargs: P.kwargs):
#         self.data = self.data.pipe(function=function, *args, **kwargs)

#         return self 
