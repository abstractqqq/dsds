import numpy as np 
import polars as pl
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def get_tp_fp(y_actual:np.ndarray, y_predicted:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Get true positive and false positive counts at thresholds.'''
    df = pl.from_records((y_predicted, y_actual), schema=["predicted", "actual"])
    all_positives = pl.lit(np.sum(y_actual))
    temp = df.lazy().groupby("predicted").agg(
        pl.col("actual").sum().alias("true_positive")
    ).sort("predicted").with_columns(
        predicted_positive = pl.arange(start=len(y_actual), end=1, step=-1)
        , tp = (all_positives - pl.col("true_positive").cumsum()).shift_and_fill(fill_value=all_positives, periods=1)
    ).select(
        pl.col("predicted")
        , pl.col("tp")
        , fp = pl.col("predicted_positive") - pl.col("tp")
    ).collect()

    # We are relatively sure that y_actual and y_predicted won't have null values.
    # So we can do temp["tp"].view() to get some more performance (3-8%). 
    # But that might confuse users.

    return temp["tp"].to_numpy(), temp["fp"].to_numpy(), temp["predicted"].to_numpy()

def roc_auc(y_actual:np.ndarray, y_predicted:np.ndarray) -> float:
    '''Return the Area Under the Curve metric for the model's predictions.''' 
    
    # This currently has difference of magnitude 1e-10 from the sklearn implementation, 
    # which is likely caused by sklearn adding zeros to the front? Not 100% sure

    y_a = y_actual.ravel()
    y_p = y_predicted.ravel()
    if len(y_a) != len(y_p):
        raise ValueError("Input y_actual and y_predicted do not have the same length.")

    uniques = np.unique(y_a)
    if uniques.size != 2:
        raise ValueError("Currently this only supports binary classification problems.")
    elif not (0 in uniques or 1 in uniques):
        raise ValueError("Currently this only supports binary classification problems with 0 and 1 target.")

    tp, fp, _ = get_tp_fp(y_a.astype(np.int8), y_p)
    return float(-np.trapz(tp/tp[0], fp/fp[0]))

def r2(y_actual:np.ndarray, y_predicted:np.ndarray) -> float:
    '''Returns the r2 of the prediction.'''

    # This is trivial, and we won't really have any performance gain by using Polars' or other stuff.
    # This is here just for completeness
    d1 = y_actual - y_predicted
    d2 = y_actual - np.mean(y_actual)
    # ss_res = d1.dot(d1), ss_tot = d2.dot(d2) 
    return 1 - d1.dot(d1)/d2.dot(d2)

def adjusted_r2(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , p:int
) -> float:
    '''Returns the adjusted r2 of the prediction.
        p: number of predictive variables
    '''
    r_squared = r2(y_actual, y_predicted)
    df_tot = len(y_actual) - 1
    return 1 - (1-r_squared) * df_tot / (df_tot - p)