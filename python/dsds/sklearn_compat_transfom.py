from typing import Union
import polars as pl
import logging
logger = logging.getLogger(__name__)
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    import pandas as pd
    import pyarrow
except Exception as e:
    logger.error(e)
    logger.error("This module requires scikit-learn, pandas and pyarrow to work.")

class PolarsExprTransformer(BaseEstimator, TransformerMixin):
    '''
    A Scikit-learn pipeline compatible Transformer for dumping Polars Expressions on a Pandas 
    dataframe. You need to import set_config from sklearn and do set_config(transform_output = "pandas")
    '''
    def __init__(self, 
        polars_exprs:Union[list[pl.Expr], pl.Expr], 
    ):
        if isinstance(polars_exprs, pl.Expr):
            e = [polars_exprs]
        elif isinstance(polars_exprs, list):
            e = [ex for ex in polars_exprs if isinstance(ex, pl.Expr)]
        else:
            raise TypeError(f"Argument `expr` must be a Polars Expression or a list of Polars Expressions, not {polars_exprs}")
        
        self.polars_exprs = e
        self.feature_names_in_:list[str] = []
        self.feature_names_out_:list[str] = []

    def get_feature_names_in(self) -> list[str]:
        return list(self.feature_names_in) # copy
    
    def get_feature_names_out(self) -> list[str]:
        return list(self.feature_names_out) # copy
    
    def fit(self, X:pd.DataFrame, y=None):
        try:
            cols = X.columns
            self.feature_names_in = cols
        except Exception as e:
            logger.error(e)
            self.feature_names_in = []
        
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        try:
            df = pl.DataFrame(X)
            out = df.with_columns(self.polars_exprs)
            self.feature_names_out = out.columns
            return out.to_pandas()
        except Exception as e:
            logger.error(e)