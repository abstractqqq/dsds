import polars as pl
import numpy as np
from pathlib import Path
from .type_alias import (
    PolarsFrame
    , ClassifModel
    # , RegressionModel
)
from typing import Optional
from .blueprint import Blueprint, Step
from dataclasses import dataclass

# --------------------- Other, miscellaneous helper functions ----------------------------------------------
@dataclass
class NumPyDataCube:
    X: np.ndarray
    y: np.ndarray
    features: list[str]
    target: str

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
        Create NumPy feature matrix X and target y from dataframe and target. 
        
        Note that this implementation will "consume df" at the end.

        Arguments:
            df:
            target:
            flatten:
            low_memory:
        returns:
            ()
        
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

def dump_blueprint(df:pl.LazyFrame, path:str|Path) -> pl.LazyFrame:
    if isinstance(df, pl.LazyFrame):
        df.blueprint.preserve(path)
        return df
    raise TypeError("Blueprints only work with LazyFrame.")

def append_classif_score(
    df: PolarsFrame
    , model:ClassifModel
    , target: Optional[str] = None
    , features: Optional[list[str]] = None
    , score_idx:int = -1 
    , score_col:str = "model_score"
) -> PolarsFrame:
    '''
    Appends a classification model to the pipeline. This step will collect the lazy frame. All non-target
    column will be used as features.

    If input df is lazy, this step will be remembered by the pipelien by default.

    Parameters
    ----------
    model
        The trained classification model
    target
        The target of the model, which will not be used in making the prediction. It is only used so that we can 
        remove it from feature list.
    features
        The features the model takes. If none, will use all non-target features.
    score_idx
        The index of the score column in predict_proba you want to append to the dataframe. E.g. -1 will take the 
        score of the positive class in a binary classification
    score_col
        The name of the score column
    '''
    if isinstance(df, pl.LazyFrame):
        return df.blueprint.add_classif(model, target, features, score_idx, score_col)
    return Blueprint._process_classif(df, model, target, features, score_idx, score_col)
    

