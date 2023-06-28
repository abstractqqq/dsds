from typing import TypeAlias, Literal, Final, Tuple
import polars as pl
import os

CPU_COUNT:Final[int] = os.cpu_count()
POLARS_NUMERICAL_TYPES:Final[Tuple[pl.DataType]] = (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)  # noqa: E501
POLARS_DATETIME_TYPES:Final[Tuple[pl.DataType]] = (pl.Datetime, pl.Date, pl.Time)

PolarsFrame:TypeAlias = pl.DataFrame | pl.LazyFrame
StepName = Literal["with_column", "map_dict", "drop", "select"]
MRMRStrategy = Literal["fscore", "f", "f_score", "xgb", "xgboost", "rf", "random_forest", "mis"
                       , "mutual_info_score", "lgbm", "lightgbm"]
ScalingStrategy = Literal["normal", "standard", "normalize", "min_max", "const", "constant"]
ImputationStrategy = Literal["mean", "avg", "average", "median", "const", "constant", "mode", "most_frequent"]
PowerTransformStrategy = Literal["yeo_johnson", "yeojohnson", "box_cox", "boxcox"]

def clean_strategy_str(s:str):
    '''Strategy strings will only have _, no -, and all lowercase.'''
    return s.strip().replace("-", "_").lower()