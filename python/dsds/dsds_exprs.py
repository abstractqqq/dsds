import polars as pl
# import numpy as np

@pl.api.register_expr_namespace("dist")
class Distance:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def abs_dist(self, to:float) -> pl.Expr:
        return (pl.lit(to) - self._expr).abs()