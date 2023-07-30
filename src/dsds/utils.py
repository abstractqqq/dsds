from typing import Optional, Union
from .blueprint import Blueprint
from pathlib import Path
from .type_alias import (
    PolarsFrame
    , ClassifModel
    , RegressionModel
)
from dataclasses import dataclass
from scipy.special import expit
import gc
import polars as pl
import numpy as np

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
        columns = []
        for c in features:
            columns.append(
                df_local.drop_in_place(c).to_numpy().reshape((-1,1))
            )
        X = np.concatenate(columns, axis=1)
    else:
        X = df_local[features].to_numpy()

    df = df.clear() # Reset to empty.
    return NumPyDataCube(X, y, features, target)

def dump_blueprint(df:pl.LazyFrame, path:Union[str,Path]) -> pl.LazyFrame:
    if isinstance(df, pl.LazyFrame):
        df.blueprint.preserve(path)
        return df
    raise TypeError("Blueprints only work with LazyFrame.")

def checkpoint(
    df:PolarsFrame
    , temp_file_path:Union[str,Path]
    , limit:int = -1
    , **kwargs) -> PolarsFrame:
    '''
    A wrapper to save a temp copy of the data so far in either parquet or csv format at the given path. This
    is purely a side effect. The difference between this and sink is that this gives you the option to save 
    the temp file as a .csv, which can be checked by a human.

    This step is pipeline compatible but will not be persisted in the blueprint.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    temp_file_path
        Path to write to. Must either end with .csv or .parquet
    limit
        Limit the number of rows to write. If <= 0, write all
    
    All other keyword arguments will be passed to Polars' write_csv or write_parquet
    '''
    ext = temp_file_path.split(".")[-1]
    if ext == "csv":
        if limit > 0:
            df.lazy().limit(limit).collect().write_csv(temp_file_path, **kwargs)
        else:
            df.lazy().collect().write_csv(temp_file_path, **kwargs)
    elif ext == "parquet":
        if limit > 0:
            df.lazy().limit(limit).collect().write_parquet(temp_file_path, **kwargs)
        else:
            df.lazy().collect().write_parquet(temp_file_path, **kwargs)
    else:
        raise ValueError(f"Temp file path must end with .csv or .parquet, not {ext}.")
    
    return df

def sink_parquet(
    df: pl.LazyFrame
    , path: Union[str,Path]
    , **kwargs
) -> pl.LazyFrame:
    '''
    A wrapper for sink_parquet so that it is pipeline compatible. This will not be persisted in the blueprint.
    '''
    df.sink_parquet(path, **kwargs)
    return df

def sink_ipc(
    df: pl.LazyFrame
    , path: Union[str,Path]
    , **kwargs
) -> pl.LazyFrame:
    '''
    A wrapper for sink_ipc so that it is pipeline compatible. This will not be persisted in the blueprint.
    '''
    df.sink_ipc(path, **kwargs)
    return df

def garbage_collect(df:PolarsFrame, persist:bool=False) -> PolarsFrame:
    '''
    A wrapper so that garbage collect can be part of the pipeline. This is purely a side effect.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    persist
        Set it to true if df is lazy and you want to persist this step in the pipeline
    '''
    _ = gc.collect()
    if isinstance(df, pl.LazyFrame) and persist:
        return df.blueprint.add_func(df, garbage_collect, {})
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
    
class Identity(ClassifModel, RegressionModel):

    def __init__(self, col:str="", idx:int=-1):
        '''
        A identity passthrough. When incoming data is a Polars DataFrame, please supply a column name.
        If incoming data is a NumPy matrix, please supply a positive index for the passthrough column.
        This is built for use in Polars pipelines. But NumPy compatibility is also considered.
        '''
        self.col = col
        self.idx = idx
    
    def fit(self, _:Union[np.ndarray,pl.DataFrame]):
        return
    
    def predict_proba(self, X:Union[np.ndarray,pl.DataFrame]) -> np.ndarray:
        if isinstance(X, pl.DataFrame) and self.col in X.columns:
            return X[self.col].to_numpy()
        elif isinstance(X, np.ndarray) and self.idx > 0:
            return X[:, self.idx]
        raise ValueError(f"Either the column {self.col} is not in X. Or X is a NumPy matrix, but idx "
                         "is not initialized. Cannot do a passthrough.")
    
    # They are the same in the identity case
    def predict(self, X:Union[np.ndarray,pl.DataFrame]) -> np.ndarray:
        return self.predict_proba(X)

def id_passthrough(
    df: PolarsFrame
    , col:str
    , as_reg: bool = False
    , score_col:str = "score"
) -> PolarsFrame:
    '''
    Appends an identity passthrough to the pipeline. This step will collect the lazy frame. 

    If input df is lazy, this step will be remembered by the blueprint by default.

    Parameters
    ----------
    df
        Either an eager or a lazy Polars DataFrame
    col
        The col that will be used as x in the linear function
    as_reg
        If true, treat this identity as a regression. Otherwise, treat it as a classification.
    score_col
        The name of the score column
    '''
    model = Identity(col=col)
    if isinstance(df, pl.LazyFrame):
        if as_reg:
            return df.blueprint.add_regression(model, [col], None, score_col)
        else:
            return df.blueprint.add_classif(model, [col], None, -1, score_col)
    else:
        if as_reg:
            return Blueprint._process_regression(df, model, [col], None, score_col)
        else:
            return Blueprint._process_classif(df, model, [col], None, -1, score_col)

class Logistic(ClassifModel):

    def __init__(self, col:str="", idx:int=-1, coeff:float = 1.0, const:float=0., k:float=1.0):
        '''
        A logistic passthrough. When incoming data is a Polars DataFrame, please supply a column name.
        If incoming data is a NumPy matrix, please supply a positive index for the passthrough column.
        This is built for use in Polars pipelines. But NumPy compatibility is also considered.

        The formula used is 1/(1 + exp(-k(coeff*x + const)))
        '''
        self.col = col
        self.idx = idx
        self.coeff = coeff
        self.const = const
        self.k = k

    def fit(self, _:Union[np.ndarray,pl.DataFrame]):
        return 
    
    def predict_proba(self, X:Union[np.ndarray,pl.DataFrame]) -> np.ndarray:
        if isinstance(X, pl.DataFrame) and self.col in X.columns:
            x = X.select(self.col).to_numpy()
        elif isinstance(X, np.ndarray) and self.idx > 0:
            x = X[:, self.idx].reshape((-1, 1))
        else:
            raise ValueError(f"Either the column {self.col} is not in X. Or X is a NumPy matrix, but idx "
                            "is not initialized. Cannot do a passthrough.")
        
        return expit(self.k * (self.coeff * x + self.const))
    
    def predict(self, X:Union[np.ndarray,pl.DataFrame]) -> np.ndarray:
        '''Predict using 0.5 as threshold.'''
        return np.round(self.predict_proba(X)).astype(np.int8)

def logistic_passthrough(
    df: PolarsFrame
    , col: str
    , coeff: float = 1.
    , const: float = 0.
    , k: float = 1.
    , score_col:str = "logistic_score"
) -> PolarsFrame:
    '''
    Appends a linear model to the pipeline. This step will collect the lazy frame.
    
    The formula used is 1/(1 + exp(-k(coeff*x + const)))

    If input df is lazy, this step will be remembered by the blueprint by default.

    Parameters
    ----------
    df
        Either an eager or a lazy Polars DataFrame
    col
        The col that will be used as x in the logistic function
    k
        The logistic scaling term
    midpoint
        The midpoint of the logistic function
    score_col
        The name of the score column
    '''
    model = Logistic(col=col, coeff=coeff, const=const, k=k)
    if isinstance(df, pl.LazyFrame):
        return df.blueprint.add_classif(model, [col], None, -1, score_col)
    return Blueprint._process_classif(df, model, [col], None, -1, score_col)

class Linear(RegressionModel):

    def __init__(self, col:str="", idx:int=-1, coeff:float=1.0, const:float=0.):
        '''
        A linear passthrough. When incoming data is a Polars DataFrame, please supply a column name.
        If incoming data is a NumPy matrix, please supply a positive index for the passthrough column.
        This is built for use in Polars pipelines. But NumPy compatibility is also considered.

        The formula is coeff * x + const
        '''
        self.col = col
        self.idx = idx
        self.coeff = coeff
        self.const = const

    def fit(self, _:Union[np.ndarray,pl.DataFrame]):
        return
    
    def predict(self, X:Union[np.ndarray,pl.DataFrame]) -> np.ndarray:
        if isinstance(X, pl.DataFrame) and self.col in X.columns:
            x = X.select(self.col).to_numpy()
        elif isinstance(X, np.ndarray) and self.idx > 0:
            x = X[:, self.idx].reshape((-1, 1))
        else:
            raise ValueError(f"Either the column {self.col} is not in X. Or X is a NumPy matrix, but idx "
                            "is not initialized. Cannot do a passthrough.")
        
        return self.coeff * x + self.const

def linear_passthrough(
    df: PolarsFrame
    , col: str
    , coeff: float = 1.
    , const: float = 0.
    , score_col:str = "linear_score"
) -> PolarsFrame:
    '''
    Appends a linear model to the pipeline. This step will collect the lazy frame. 
    
    The formula is coeff * x + const

    If input df is lazy, this step will be remembered by the blueprint by default.

    Parameters
    ----------
    df
        Either an eager or a lazy Polars DataFrame
    col
        The col that will be used as x in the linear function
    coeff
        The coefficient of the linear function
    const
        The constant term of the linear function
    score_col
        The name of the score column
    '''
    model = Linear(col=col, coeff=coeff, const=const)
    if isinstance(df, pl.LazyFrame):
        return df.blueprint.add_regression(model, [col], None, score_col)
    return Blueprint._process_regression(df, model, [col], None, score_col)