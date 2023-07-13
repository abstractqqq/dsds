from typing import (
    TypeAlias
    , Literal
    , Final
    , Tuple
    , Callable
    , Concatenate
    , ParamSpec
)
import polars as pl
import os

P = ParamSpec('P')
CPU_COUNT:Final[int] = os.cpu_count()
POLARS_NUMERICAL_TYPES:Final[Tuple[pl.DataType]] = (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)  # noqa: E501
POLARS_DATETIME_TYPES:Final[Tuple[pl.DataType]] = (pl.Datetime, pl.Date, pl.Time)

PolarsFrame:TypeAlias = pl.DataFrame | pl.LazyFrame
PipeFunction = Callable[Concatenate[PolarsFrame, P], PolarsFrame]
ActionType = Literal["with_columns", "map_dict", "drop", "select", "apply_func", "filter"]
MRMRStrategy = Literal["fscore", "f", "f_score", "xgb", "xgboost", "rf", "random_forest", "mis"
                       , "mutual_info_score", "lgbm", "lightgbm"]
ScalingStrategy = Literal["normal", "standard", "normalize", "min_max", "const", "constant"]
ImputationStrategy = Literal["mean", "avg", "average", "median", "const", "constant", "mode", "most_frequent"]
PowerTransformStrategy = Literal["yeo_johnson", "yeojohnson", "box_cox", "boxcox"]
KSAlternatives = Literal["two-sided", "greater", "less"]

BinaryModels = Literal["logistic", "lr", "lightgbm", "lgbm", "xgboost", "xgb", "random_forest", "rf"]
# This is just a subset of Scipy.stats's distributions which can be named by strings. All scipy.stats's string-name-able
# distributions should work when the arguments asks for a CommonContinuousDist.
CommonContinuousDist = Literal["norm", "lognorm", "truncnorm", "uniform", "t", "beta", "cauchy", "expon", "gamma"]
def clean_strategy_str(s:str):
    '''Strategy strings will only have _, no -, and all lowercase.'''
    return s.strip().replace("-", "_").lower()