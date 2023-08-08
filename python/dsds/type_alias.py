from typing import (
    Literal
    , Final
    , Tuple
    , Union
    , Callable
)
from abc import ABC, abstractmethod
import polars as pl
import sys
if sys.version_info >= (3, 10):
    from typing import TypeAlias, Concatenate, ParamSpec
    P = ParamSpec('P')
    PolarsFrame:TypeAlias = Union[pl.DataFrame, pl.LazyFrame]
    PipeFunction = Callable[Concatenate[PolarsFrame, P], PolarsFrame]
else: # 3.9
    from typing_extensions import TypeAlias
    PolarsFrame:TypeAlias = Union[pl.DataFrame, pl.LazyFrame]
    PipeFunction = Callable

import os
import numpy as np

# --- Constants ---
CPU_COUNT:Final[int] = os.cpu_count()
CPU_M1: Final[int] = CPU_COUNT - 1
POLARS_NUMERICAL_TYPES:Final[Tuple[pl.DataType]] = (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64
                                                    , pl.Int8, pl.Int16, pl.Int32, pl.Int64)
POLARS_DATETIME_TYPES:Final[Tuple[pl.DataType]] = (pl.Datetime, pl.Date)

# --- Strategies ---
MRMRStrategy:TypeAlias = Literal["fscore", "f", "f_score", "xgb", "xgboost", "rf", "random_forest", "mis"
                                , "mutual_info_score", "lgbm", "lightgbm"]
ScalingStrategy:TypeAlias = Literal["standard", "min_max", "const", "constant"]
ImputationStrategy:TypeAlias = Literal["mean", "avg", "average", "median", "const", "constant", "mode", "most_frequent"]
PowerTransformStrategy:TypeAlias = Literal["yeo_johnson", "yeojohnson", "box_cox", "boxcox"]
WeightStrategy: TypeAlias = Literal["none", "balanced", "custom"]
ZeroOneCombineStrategy = Literal["union", "intersection", "same"]

# --- Models ---
BinaryModels:TypeAlias = Literal["logistic", "lr", "lightgbm", "lgbm", "xgboost", "xgb", "random_forest", "rf"]

# --- Extracts ---
DateExtract:TypeAlias = Literal["year", "quarter", "month", "week", "day_of_week", "day_of_year"]
ListExtract:TypeAlias = Literal["min", "max", "mean", "len", "first", "last"]
HorizontalExtract:TypeAlias = Literal["min", "max", "sum", "any", "all"]
StrExtract: TypeAlias = Literal["count", "len", "contains", "starts_with", "ends_with"]

# --- Stats Related ---
Alternatives = Literal["two-sided", "greater", "less"]
# This is just a subset of Scipy.stats's distributions which can be named by strings. 
# All scipy.stats's string-name-able distributions should work when the arguments asks 
# for a CommonContinuousDist.
CommonContiDist:TypeAlias = Literal["norm", "lognorm", "truncnorm", "uniform", "t", "beta", "cauchy", "expon", "gamma"]

# --- Blueprint ---
ActionType:TypeAlias = Literal["with_columns", "map_dict", "select", "drop", "add_func", "filter", "classif"
                               , "regression"]
# --- Other ---
SimpleDtypes:TypeAlias = Literal["numeric", "datetime", "bool", "list", "string", "other/unknown"]
InnerDtypes:TypeAlias = Literal["int", "str"]
Stemmer:TypeAlias = Literal["snowball", ""]


# --- Utils ---
def clean_strategy_str(s:str):
    '''Strategy strings will only have _, no -, and all lowercase.'''
    return s.strip().replace("-", "_").lower()

# --- ABC ---
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