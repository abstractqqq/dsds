from typing import Optional, Union
from .blueprint import (
    Blueprint,
    _dsds_with_columns
)
from pathlib import Path
from .type_alias import (
    PolarsFrame
    , ClassifModel
    , RegressionModel
)
from dsds._rust import rs_gcc_proba_est
from dataclasses import dataclass
import polars as pl
import numpy as np
import gc

# --------------------- Other, miscellaneous helper functions ----------------------------------------------
@dataclass
class NumPyDataCube:
    X:np.ndarray
    y:np.ndarray
    features:list[str]
    target:str

    def to_df(self) -> pl.DataFrame:
        if self.X.shape[0] != len(self.y.ravel()):
            raise ValueError("NumPyDataCube's X and y must have the same number of rows.") 

        df = pl.from_numpy(self.X, schema=self.features)
        t = pl.Series(self.target, self.y)
        return df.insert_at_idx(0, t)

def get_numpy(
    df:PolarsFrame
    , target:str
    , flatten:bool=True
    , low_memory:bool=True
) -> NumPyDataCube:
    '''
    Create NumPy feature matrix X and target y from dataframe and target. Note that this implementation will 
    "clear df" at the end.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    target
        The name of the target column
    flatten
        Whether to flatten target or not
    low_memory
        Whether to do NumPy conversion column by column or all at once     
    '''
    features:list[str] = df.columns 
    features.remove(target)
    df_local = df.lazy().collect()
    y = df_local.drop_in_place(target).to_numpy()
    if flatten:
        y = y.ravel()
    
    if low_memory:
        columns = [
            df_local.drop_in_place(c).to_numpy().reshape((-1,1))
            for c in features
        ]
        X = np.concatenate(columns, axis=1)
    else:
        X = df_local[features].to_numpy()

    return NumPyDataCube(X, y, features, target)

def dump_blueprint(df:pl.LazyFrame, path:Union[str,Path]) -> pl.LazyFrame:
    if isinstance(df, pl.LazyFrame):
        df.blueprint.preserve(path)
        return df
    raise TypeError("Blueprints only work with LazyFrame.")

def garbage_collect(df:PolarsFrame) -> PolarsFrame:
    '''
    A wrapper so that garbage collect can be part of the pipeline. This is purely a side effect.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    '''
    _ = gc.collect()
    return df

def append_classif_score(
    df: PolarsFrame
    , model:ClassifModel
    , features: list[str]
    , target: Optional[str] = None
    , score_idx:int = -1 
    , score_col:str = "model_score"
) -> PolarsFrame:
    '''
    Appends a classification model to the pipeline. This step will collect the lazy frame.

    If input df is lazy, this step will be remembered by the blueprint by default.

    Parameters
    ----------
    model
        The trained classification model
    features
        The features the model takes
    target
        The target of the model, which will not be used in making the prediction. It is only here so that 
        we can remove it from feature list if it is in features by mistake.
    score_idx
        The index of the score column in predict_proba you want to append to the dataframe. E.g. -1 will take the 
        score of the positive class in a binary classification
    score_col
        The name of the score column
    '''
    if isinstance(df, pl.LazyFrame):
        return df.blueprint.add_classif(model, features, target, score_idx, score_col)
    return Blueprint._process_classif(df, model, features, target, score_idx, score_col)

def append_regression(
    df: PolarsFrame
    , model:RegressionModel
    , features: list[str]
    , target: Optional[str] = None
    , score_col:str = "model_score"
) -> PolarsFrame:
    '''
    Appends a regression model to the pipeline. This step will collect the lazy frame.

    If input df is lazy, this step will be remembered by the blueprint by default.

    Parameters
    ----------
    model
        The trained classification model
    features
        The features the model takes
    target
        The target of the model, which will not be used in making the prediction. It is only here so that we can 
        remove it from feature list if it is in features by mistake.
    score_col
        The name of the score column
    '''
    if isinstance(df, pl.LazyFrame):
        return df.blueprint.add_regression(model, features, target, score_col)
    return Blueprint._process_regression(df, model, features, target, score_col)
    
    
def id_passthrough(
    df: PolarsFrame
    , col:str
    , suffix:str = "_id_passthrough"
) -> PolarsFrame:
    '''
    Appends an identity passthrough to the pipeline.

    This step will be remembered by the blueprint by default.
    '''
    return _dsds_with_columns(df, [pl.col(col).suffix(suffix)])

def logistic_passthrough(
    df: PolarsFrame
    , col:str
    , coeff:float = 1.0
    , const:float = 0.
    , k:float = 1.0
    , suffix:str = "_logistic"
) -> PolarsFrame:
    '''
    Appends a logistic regression passthrough to the pipeline. The formula used is 
    1/(1 + exp(-k(coeff*x + const))).

    This step will be remembered by the blueprint by default.
    '''
    expr = (pl.lit(1.0)/(pl.lit(1.0) + (pl.lit(-k) * (pl.col(col)*pl.lit(coeff) + pl.lit(const))).exp()))\
            .alias(f"{col}{suffix}")
    return _dsds_with_columns(df, [expr])

def linear_passthrough(
    df: PolarsFrame
    , col:str
    , coeff:float = 1.0
    , const:float = 0.
    , suffix:str = "_linear"
) -> PolarsFrame:
    '''
    Appends a linear passthrough to the pipeline. The formula is coeff * x + const.

    This step will be remembered by the blueprint by default.
    '''
    return _dsds_with_columns(df, [(pl.col(col)*pl.lit(coeff) + pl.lit(const)).suffix(suffix)])

# ----------------------------------------------------------------------------------------------------------------------

def gcc_proba_est(
    total: int,
    subsample_amt:int,
    n_times: int = 1,
    start_at: int = 10,
    threshold:float = 0.9,
    n_trials: int = 1000
) -> int:
    '''
    Estimates the probablity of a Generalized Coupon Collector's problem:
    There are `total` number of coupons, say each time we are subsampling `subsample_amt` of them. How many
    times do we need to subsample (`subsample_times`) so that we are at least `threshold` sure that we 
    will see all coupons for at least `n_times`?

    The algorithm will run a Monte-Carlo simulation with `n_trials` to estimate the probablity and increment 
    `subsample_times` until the estimated probability >= threshold. It is not yet fully optimized.

    Notice that the minimum `subsample_times` the algorithm will return is always `start_at`, but in reality 
    this number may be smaller. It is designed this way so that for larger input values, the user can supply
    a heuristic for a lower bound and the algorithm can get a head start and converge faster.
    '''
    return rs_gcc_proba_est(n_trials, total, subsample_amt, n_times, start_at, threshold)