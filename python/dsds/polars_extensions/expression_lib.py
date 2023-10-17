# import math
# import numpy as np
import polars as pl
from typing import Union
from polars.utils.udfs import _get_shared_lib_location

lib = _get_shared_lib_location(__file__)

_BENFORD_DIST_SERIES = (1 + 1 / pl.int_range(1, 10, eager=True)).log10()

@pl.api.register_expr_namespace("dsds_num")
class NumExt:
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
    
    def gcd(self, other:Union[int, pl.Expr]) -> pl.Expr:
        '''
        Computes GCD of two integer columns.

        Parameters
        ----------
        other
            Either an int or a Polars expression
        '''
        if isinstance(other, int):
            other_ = pl.lit(other, dtype=pl.Int64)
        else:
            other_ = other.cast(pl.Int64)
        
        return self._expr.cast(pl.Int64)._register_plugin(
            lib=lib,
            symbol="pl_gcd",
            args = [other_],
            is_elementwise=True,
        )
    
    def lcm(self, other:Union[int, pl.Expr]) -> pl.Expr:
        '''
        Computes LCM of two integer columns.

        Parameters
        ----------
        other
            Either an int or a Polars expression
        '''
        if isinstance(other, int):
            other_ = pl.lit(other, dtype=pl.Int64)
        else:
            other_ = other.cast(pl.Int64)
        
        return self._expr.cast(pl.Int64)._register_plugin(
            lib=lib,
            symbol="pl_lcm",
            args = [other_],
            is_elementwise=True,
        )
    
    def lempel_ziv_complexity(self, as_ratio:bool=False) -> pl.Expr:
        '''
        Computes the Lempel Ziv Complexity. Input must be a boolean column.

        Parameters
        ----------
        as_ratio
            If true, will return the complexity / the length of the column
        '''
        out = self._expr._register_plugin(
            lib=lib,
            symbol="pl_lempel_ziv_complexity",
            is_elementwise=True,
        )
        if as_ratio:
            return out / self._expr.len()
        return out

    
@pl.api.register_expr_namespace("dsds_str")
class StrExt:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def str_jaccard(
        self,
        other: Union[str, pl.Expr],
        substr_size: int = 2
    ) -> pl.Expr:
        '''
        Treats each string as a set. Duplicate chars will be removed. And computes the jaccard similarity between
        this word and the other.
        
        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this 
            is an expression, then an element-wise jaccard similarity computation between this column 
            and the other (given by the expression) will be performed.
        substr_size 
            The substring size for Jaccard similarity. E.g. if substr_size = 2, "apple" will be decomposed into
            the set ('ap', 'pp', 'pl', 'le') before being compared.
        '''
        if isinstance(other, str):
            other_ = pl.lit(other)
        else:
            other_ = other
        
        return self._expr._register_plugin(
            lib=lib,
            symbol="pl_str_jaccard",
            args = [other_, pl.lit(substr_size, dtype=pl.UInt32)],
            is_elementwise=True,
        )
    
    def levenshtein_dist(
        self, 
        other:Union[str, pl.Expr], 
    ) -> pl.Expr:
        '''
        Computes the levenshtein distance between this each value in the column with the str other.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this 
            is an expression, then an element-wise Levenshtein distance computation between this column 
            and the other (given by the expression) will be performed.
        '''
        if isinstance(other, str):
            other_ = pl.lit(other)
        else:
            other_ = other

        return self._expr._register_plugin(
            lib=lib,
            symbol="pl_levenshtein_dist",
            args = [other_],
            is_elementwise=True,
        )
    
    def hamming_dist(
        self, 
        other:Union[str, pl.Expr], 
    ) -> pl.Expr:
        '''
        Computes the hamming distance between two strings. If they do not have the same length, null will
        be returned.

        Parameters
        ----------
        other
            If this is a string, then the entire column will be compared with this string. If this 
            is an expression, then an element-wise hamming distance computation between this column 
            and the other (given by the expression) will be performed.
        '''
        if isinstance(other, str):
            other_ = pl.lit(other)
        else:
            other_ = other

        return self._expr._register_plugin(
            lib=lib,
            symbol="pl_hamming_dist",
            args = [other_],
            is_elementwise=True,
        )
    
    def snowball(self) -> pl.Expr:
        '''
        Applies the snowball stemmer for the column. The column is supposed to be a column of single words.
        '''
        return self._expr._register_plugin(
            lib=lib,
            symbol="pl_snowball_stem",
            is_elementwise=True,
        )