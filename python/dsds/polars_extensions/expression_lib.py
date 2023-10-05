# import math
# import numpy as np
import polars as pl
from polars.utils.udfs import _get_shared_lib_location

lib = _get_shared_lib_location(__file__)

_BENFORD_DIST_SERIES = (1 + 1 / pl.int_range(1, 10, eager=True)).log10()

@pl.api.register_expr_namespace("dsds_exprs")
class DSDSExprs:
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
    
    def cv(self, ddof:int = 1) -> pl.Expr:
        '''
        Returns the coefficient of variation of the expression
        '''
        return self._expr.std(ddof=ddof) / self._expr.mean()
    
    def z_normalize(self, ddof:int=1) -> pl.Expr:
        '''
        z_normalize the given expression: remove the mean and scales by the std
        '''
        return (self._expr - self._expr.mean()) / self._expr.std(ddof=ddof)
    
    def benford_correlation(self) -> pl.Expr:
        '''
        Returns the benford correlation for the given expression.
        '''
        counts = (
            # This when then is here because there is a precision issue that happens for 1000.
            pl.when(self._expr.abs() == 1000).then(
                pl.lit(1)
            ).otherwise(
                (self._expr.abs()/(pl.lit(10).pow((self._expr.abs().log10()).floor())))
            ).drop_nans()
            .drop_nulls()
            .cast(pl.UInt8)
            .append(pl.int_range(1, 10, eager=False))
            .value_counts()
            .sort()
            .struct.field("counts") - pl.lit(1)
        )
        return pl.corr(counts, pl.lit(_BENFORD_DIST_SERIES))
    
    def frac(self) -> pl.Expr:
        '''
        Returns the fractional part of the input values. E.g. fractional part of 1.1 is 0.1
        '''
        return self._expr.mod(1.0)
    
    def levenshtein_dist(self, other:str, lowercase:bool=False) -> pl.Expr:
        '''
        Computes the levenshtein distance between this each value in the column with the str other.

        Parameters
        ----------
        other
            The string to compare with
        lowercase
            Whether to lowercase all strings
        '''
        if lowercase:
            other_str = pl.Series([other.lower()])
            expr = self._expr.str.to_lowercase()
        else:
            other_str = pl.Series([other])
            expr = self._expr

        return expr._register_plugin(
            lib=lib,
            symbol="pl_levenshtein_dist",
            args = [other_str],
            is_elementwise=True,
        )
    
    def snowball(self, lowercase:bool=False) -> pl.Expr:
        '''
        Applies the snowball stemmer for the column. The column is supposed to be a column of single words.

        Parameters
        ----------
        lowercase
            Whether to lowercase all strings
        '''
        if lowercase:
            expr = self._expr.str.to_lowercase()
        else:
            expr = self._expr

        return expr._register_plugin(
            lib=lib,
            symbol="pl_snowball_stem",
            is_elementwise=True,
        )