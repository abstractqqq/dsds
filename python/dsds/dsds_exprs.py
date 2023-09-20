import polars as pl
# from polars.utils.udfs import _get_shared_lib_location

# lib = _get_shared_lib_location(__file__)
# import numpy as np

@pl.api.register_expr_namespace("dsds_exprs")
class MoreExprs:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def abs_diff(self, to:float) -> pl.Expr:
        '''
        Returns the absolute difference between the expression and the value `to`
        '''
        return (pl.lit(to) - self._expr).abs()
    
    def harmonic_mean(self) -> pl.Expr:
        '''
        Returns the harmonic mean of the expression
        '''
        return (
            self._expr.count() / (pl.lit(1.0) / self._expr).sum()
        )
    
    def rms(self) -> pl.Expr: 
        '''
        Returns root mean square of the expression
        '''
        return (self._expr.dot(self._expr)/self._expr.count()).sqrt()
    
    def cov(self, ddof:int = 1) -> pl.Expr:
        '''
        Returns the coefficient of variation of the expression
        '''
        return self._expr.std(ddof=ddof) / self._expr.mean()
    
    def z_normalize(self) -> pl.Expr:
        '''
        z_normalize the given expression: remove the mean and scales by the std(ddof=1)
        '''
        return (self._expr - self._expr.mean()) / self._expr.std()

    # def levenshtein_dist(self, ref:str) -> pl.Expr:
    #     return self._expr._register_plugin(
    #         lib=lib,
    #         symbol="levenshtein_dist",
    #         args = [ref],
    #         is_elementwise=True,
    #     )