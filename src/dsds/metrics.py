import numpy as np 
import polars as pl
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# No need to do length checking (len(y_1) == len(y_2)) because NumPy / Polars will complain for us.
# Everything here is essentially fun stuff.
# Ideally speaking, you should never convert things to dataframes when you compute these metrics.
# But using only NumPy means we loss parallelism, and using Python concurrent module will not help
# in this case (haha Python). 
# So funnily enough, we can convert things to dataframes and get a performance boost.
# Ideally, all of these should be re-written in Rust using some kind of parallel stuff in Rust.



def _flatten_input(y_actual: np.ndarray, y_predicted:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_a = y_actual.ravel()
    if y_predicted.ndim == 2:
        y_p = y_predicted[:, 1] # .ravel()
    else:
        y_p = y_predicted.ravel()

    return y_a, y_p

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
    '''Return the Area Under the Curve metric for the model's predictions.

        Arguments:
            y_actual: actual target
            y_predicted: predicted probability

        Returns:
            the auc value
    ''' 
    
    # This currently has difference of magnitude 1e-10 from the sklearn implementation, 
    # which is likely caused by sklearn adding zeros to the front? Not 100% sure
    # This is about 50% faster than sklearn's implementation. I know, not that this matters
    # that much...
    
    y_a, y_p = _flatten_input(y_actual, y_predicted)
    # No need to check if length matches because Polars will complain for us
    uniques = np.unique(y_a)
    if uniques.size != 2:
        raise ValueError("Currently this only supports binary classification.")
    if not (0 in uniques and 1 in uniques):
        raise ValueError("Currently this only supports binary classification with 0 and 1 target.")

    tp, fp, _ = get_tp_fp(y_a.astype(np.int8), y_p)
    return float(-np.trapz(tp/tp[0], fp/fp[0]))

def logloss(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , sample_weights:Optional[np.ndarray]=None
    , min_prob:float = 1e-12
) -> float:
    '''Return the logloss of the prediction.

        Arguments:
            y_actual: actual target
            y_predicted: predicted probability
            sample_weights: an array of size (n_sample, ) which provides weights to each sample
            min_prob: minimum probability to clip so that we can prevent illegal computations like 
            log(0). If p < min_prob, log(min_prob) will be computed instead.

        Returns:
            the average logloss

    '''
    # Takes about 1/3 time of sklearn's log_loss because we parallelized some computations

    y_a, y_p = _flatten_input(y_actual, y_predicted)
    uniques = np.unique(y_a)
    if uniques.size != 2:
        raise ValueError("Currently this only supports binary classification.")
    if not (0 in uniques and 1 in uniques):
        raise ValueError("Currently this only supports binary classification with 0 and 1 target.")

    if sample_weights is None:
        return pl.from_records((y_a, y_p), schema=["y", "p"]).with_columns(
            l = pl.col("p").clip_min(min_prob).log(),
            o = (1- pl.col("p")).clip_min(min_prob).log(),
            ny = 1 - pl.col("y")
        ).select(
            pl.lit(-1, dtype=pl.Float64) * (pl.col("y") * pl.col("l") + pl.col("ny") * pl.col("o")).mean()
        ).row(0)[0]
    else:
        s = sample_weights.ravel()
        return pl.from_records((y_a, y_p, s), schema=["y", "p", "s"]).with_columns(
            l = pl.col("p").clip_min(min_prob).log(),
            o = (1- pl.col("p")).clip_min(min_prob).log(),
            ny = 1 - pl.col("y")
        ).select(
            pl.lit(-1, dtype=pl.Float64)
            * pl.col("s") 
            * (pl.col("y") * pl.col("l") + pl.col("ny") * pl.col("o")).mean()
        ).row(0)[0]


def l2_loss(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , sample_weights:Optional[np.ndarray]=None
) -> float:
    diff = y_actual - y_predicted
    if sample_weights is None:
        return np.mean(diff.dot(diff))
    else:
        return (sample_weights/(len(diff))).dot(diff.dot(diff))
    
mse = l2_loss

def l1_loss(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , sample_weights:Optional[np.ndarray]=None
) -> float:
    diff = np.abs(y_actual - y_predicted)
    if sample_weights is None:
        return np.mean(diff)
    else:
        return (sample_weights/(len(diff))).dot(diff)

mae = l1_loss 

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
    '''Computes the adjusted r2 of the prediction.

        p: number of predictive variables
    '''
    r_squared = r2(y_actual, y_predicted)
    df_tot = len(y_actual) - 1
    return 1 - (1-r_squared) * df_tot / (df_tot - p)

def huber_loss(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , delta:float
    , sample_weights:Optional[np.ndarray]=None  
) -> float:
    
    y_a = y_actual.ravel()
    y_p = y_predicted.ravel()
    
    if delta <= 0:
        raise ValueError("Delta in Huber loss must be positive.")

    abs_diff = np.abs(y_a - y_p)
    mask = abs_diff <= delta
    not_mask = ~mask
    loss = np.zeros(shape=abs_diff.shape)
    loss[mask] = 0.5 * (abs_diff[mask]**2)
    loss[not_mask] = delta * (abs_diff[not_mask] - 0.5 * delta)

    if sample_weights is None:
        return np.mean(loss)
    else:
        return (sample_weights/(len(loss))).dot(loss)
