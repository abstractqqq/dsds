import polars as pl
import numpy as np 
from dataclasses import dataclass
# from .eda_prescreen import *



# --------------------- Other, miscellaneous helper functions ----------------------------------------------
@dataclass
class NumPyDataCube:
    X: np.ndarray
    y: np.ndarray
    features: list[str]

def get_numpy(df:pl.DataFrame, target:str, flatten:bool=True) -> NumPyDataCube:
    '''
        Create NumPy matrices/array for feature matrix X and target y. 
        
        Note that this implementation will "consume df" column by column, thus saving memory used in the process.
        If memory is not a problem, you can do directly df.select(feature).to_numpy(). 
        IMPORTANT: df will be consumed at the end.

        Arguments:
            df:
            target:
            flatten:

        returns:
            ()
        
    '''
    features:list[str] = df.columns 
    features.remove(target)
    y = df.drop_in_place(target).to_numpy()
    if flatten:
        y = y.ravel() 
    columns = []
    for c in features:
        columns.append(
            df.drop_in_place(c).to_numpy().reshape((-1,1))
        )

    del df
    X = np.concatenate(columns, axis=1)
    return NumPyDataCube(X, y, features)


