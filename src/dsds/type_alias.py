from typing import (
    Literal
    , Final
    , Tuple
    , Union
)

import polars as pl
import sys
if sys.version_info >= (3, 10):
    from typing import TypeAlias, Concatenate, ParamSpec, Callable
    P = ParamSpec('P')
    PolarsFrame:TypeAlias = Union[pl.DataFrame, pl.LazyFrame]
    PipeFunction = Callable[Concatenate[PolarsFrame, P], PolarsFrame]
else:
    from typing_extensions import TypeAlias
    PolarsFrame:TypeAlias = Union[pl.DataFrame, pl.LazyFrame]
    PipeFunction = Callable

import os
import numpy as np
from abc import ABC, abstractmethod

CPU_COUNT:Final[int] = os.cpu_count()
CPU_M1: Final[int] = CPU_COUNT - 1
POLARS_NUMERICAL_TYPES:Final[Tuple[pl.DataType]] = (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64
                                                    , pl.Int8, pl.Int16, pl.Int32, pl.Int64)
POLARS_DATETIME_TYPES:Final[Tuple[pl.DataType]] = (pl.Datetime, pl.Date)

ActionType:TypeAlias = Literal["with_columns", "map_dict", "drop", "select", "add_func", "filter", "classif"
                               , "regression"]
MRMRStrategy:TypeAlias = Literal["fscore", "f", "f_score", "xgb", "xgboost", "rf", "random_forest", "mis"
                                , "mutual_info_score", "lgbm", "lightgbm"]
ScalingStrategy:TypeAlias = Literal["standard", "min_max", "const", "constant"]
ImputationStrategy:TypeAlias = Literal["mean", "avg", "average", "median", "const", "constant", "mode", "most_frequent"]
PowerTransformStrategy:TypeAlias = Literal["yeo_johnson", "yeojohnson", "box_cox", "boxcox"]
Alternatives = Literal["two-sided", "greater", "less"]

SimpleDtypes:TypeAlias = Literal["numeric", "datetime", "bool", "string", "other/unknown"]
BinaryModels:TypeAlias = Literal["logistic", "lr", "lightgbm", "lgbm", "xgboost", "xgb", "random_forest", "rf"]
DateExtract:TypeAlias = Literal["year", "quarter", "month", "week", "day_of_week", "day_of_year"]
ListExtract:TypeAlias = Literal["min", "max", "mean", "len", "first", "last"]

# This is just a subset of Scipy.stats's distributions which can be named by strings. All scipy.stats's string-name-able
# distributions should work when the arguments asks for a CommonContinuousDist.
CommonContinuousDist:TypeAlias = Literal["norm", "lognorm", "truncnorm", "uniform", "t", "beta", "cauchy", "expon", "gamma"]

def clean_strategy_str(s:str):
    '''Strategy strings will only have _, no -, and all lowercase.'''
    return s.strip().replace("-", "_").lower()

class ClassifModel(ABC):

    @abstractmethod
    def predict(self, X:Union[np.ndarray,pl.DataFrame]) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray,pl.DataFrame]) -> np.ndarray:
        ...

    @abstractmethod
    def fit(self, X:Union[np.ndarray,pl.DataFrame], y:Union[np.ndarray,pl.Series,pl.DataFrame]): # Should return self
        ...
    
class RegressionModel(ABC):
    @abstractmethod
    def predict(self, X: Union[np.ndarray,pl.DataFrame]) -> np.ndarray:
        ...

    @abstractmethod
    def fit(self, X:Union[np.ndarray,pl.DataFrame], y:Union[np.ndarray,pl.Series,pl.DataFrame]): # Should return self
        ...