import polars as pl
import numpy as np 
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

def get_numpy(df:pl.DataFrame, target:str, flatten:bool=True, low_memory:bool=True) -> NumPyDataCube:
    '''
        Create NumPy feature matrix X and target y from dataframe and target. 
        
        Note that this implementation will "consume df" column by column, thus saving memory used in the process.
        If memory is not a problem, you can do directly df.select(feature).to_numpy(). 

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
    y = df.drop_in_place(target).to_numpy()
    if flatten:
        y = y.ravel()
    
    if low_memory:
        columns = []
        for c in features:
            columns.append(
                df.drop_in_place(c).to_numpy().reshape((-1,1))
            )
        X = np.concatenate(columns, axis=1)
    else:
        X = df[features].to_numpy()

    df = pl.DataFrame() # Reset to empty.
    return NumPyDataCube(X, y, features, target)


# Turns zip9 into standard 5 digit zipcodes.
def clean_zip_codes():
    pass


