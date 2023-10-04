import math
import polars as pl
from polars.utils.udfs import _get_shared_lib_location

print(__file__)
lib = _get_shared_lib_location(__file__)
print(lib)
# import numpy as np

_BENFORD_DIST_SERIES = (1 + 1 / pl.int_range(1, 10, eager=True)).log10()
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
            .sort()
            .value_counts()
            .struct.field("counts") - pl.lit(1)
        )
        return pl.corr(counts, pl.lit(_BENFORD_DIST_SERIES))
    
    def frac(self) -> pl.Expr:
        '''
        Returns the fractional part of the input values. E.g. fractional part of 1.1 is 0.1
        '''
        return self._expr.mod(1.0)
    
    # def sine_wave(self, freq:float, a:float) -> pl.Expr:
    #     '''
    #     The column will be considered `time`, and this will perform the sine wave transform on this column

    #     Parameters
    #     ----------
    #     freq
    #         The frequency of the sine wave transform
    #     a
    #         The amplitude of the sine wave transform
    #     '''
    #     return pl.lit(a) * (pl.lit(2*math.pi*freq) * self._expr).sin()
    
    # def square_wave(self, a:float, shift:float = 0.) -> pl.Expr:
    #     '''
    #     The column will be considered `time`, and this will perform the square wave transform on this column

    #     Parameters
    #     ----------
    #     a
    #         The amplitude of the square wave transform
    #     shift
    #         A constant to add to the square_wave. If shift != 0, it is also called DC-shifted square wave.
    #     '''
    #     return pl.when(self._expr.mod(1.0) < 0.5).then(a).otherwise(-a) + pl.lit(shift)
    
    # def modified_sine_wave(self, freq:float, a: float) -> pl.Expr:
    #     '''
    #     The column will be considered `time`, and this will perform the modified sine wave transform on this column

    #     Parameters
    #     ----------
    #     freq
    #         The amplitude of the sine wave transform
    #     a
    #         The amplitude of the sine wave transform
    #     '''
    #     y = self._expr.mod(1.0)
    #     return pl.when(y < 0.25).then(0.).when(y < 0.5).then(a).when(y < 0.75).then(0).otherwise(-a)



    # def levenshtein_dist(self) -> pl.Expr:
    #     return self._expr._register_plugin(
    #         lib=lib,
    #         symbol="levenshtein_dist",
    #         is_elementwise=True,
    #     )