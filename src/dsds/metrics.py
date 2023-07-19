import numpy as np 
import polars as pl
from typing import Tuple, Optional, Union
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

def get_tp_fp(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , ratio:bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Get true positive and false positive counts at various thresholds.

    Parameters
    ----------
    y_actual
        Actual binary labels
    y_predicted
        Predicted probabilities
    ratio
        If true, return true positive rate and false positive rate at the threholds; if false return the count
    '''
   
    df = pl.from_records((y_predicted, y_actual), schema=["predicted", "actual"])
    all_positives = pl.lit(np.sum(y_actual))
    n = len(df)
    temp = df.lazy().groupby("predicted").agg(
        pl.count().alias("cnt")
        , pl.col("actual").sum().alias("true_positive")
    ).sort("predicted").with_columns(
        predicted_positive = n - pl.col("cnt").cumsum() + pl.col("cnt")
        , tp = (all_positives - pl.col("true_positive").cumsum()).shift_and_fill(fill_value=all_positives, periods=1)
    ).select(
        pl.col("predicted")
        , pl.col("tp")
        , fp = pl.col("predicted_positive") - pl.col("tp")
    ).collect()

    # We are relatively sure that y_actual and y_predicted won't have null values.
    # So we can do temp["tp"].view() to get some more performance. 
    # But that might confuse users.
    tp = temp["tp"].to_numpy()
    fp = temp["fp"].to_numpy()
    if ratio:
        return tp/tp[0], fp/fp[0], temp["predicted"].to_numpy()
    return tp, fp, temp["predicted"].to_numpy()

def roc_auc(y_actual:np.ndarray, y_predicted:np.ndarray, check_binary:bool=True) -> float:
    '''
    Return the Area Under the Curve metric for the model's predictions.

    Parameters
    ----------
    y_actual
        Actual binary labels
    y_predicted
        Predicted probabilities
    check_binary
        If true, checks if y_actual is binary
    ''' 
    
    # This currently has difference of magnitude 1e-10 from the sklearn implementation, 
    # which is likely caused by sklearn adding zeros to the front? Not 100% sure
    # This is about 50% faster than sklearn's implementation. I know, not that this matters
    # that much...
    
    y_a, y_p = _flatten_input(y_actual, y_predicted)
    # No need to check if length matches because Polars will complain for us
    if check_binary:
        uniques = np.unique(y_a)
        if uniques.size != 2:
            raise ValueError("Currently this only supports binary classification.")
        if not (0 in uniques and 1 in uniques):
            raise ValueError("Currently this only supports binary classification with 0 and 1 target.")

    tpr, fpr, _ = get_tp_fp(y_a.astype(np.int8), y_p, ratio=True)
    return float(-np.trapz(tpr, fpr))

def logloss(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , sample_weights:Optional[np.ndarray]=None
    , min_prob:float = 1e-12
    , check_binary:bool = True
) -> float:
    '''
    Return the logloss of the binary classification.

    Parameters
    ----------
    y_actual
        Actual binary labels
    y_predicted
        Predicted probabilities
    sample_weights
        An array of size (n_sample, ) which provides weights to each sample
    min_prob
        Minimum probability to clip so that we can prevent illegal computations like 
        log(0). If p < min_prob, log(min_prob) will be computed instead.
    '''
    # Takes about 1/3 time of sklearn's log_loss because we parallelized some computations
    y_a, y_p = _flatten_input(y_actual, y_predicted)
    if check_binary:
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
    
def binary_psi(
    new_score: Union[pl.Series, np.ndarray]
    , old_score: Union[pl.Series, np.ndarray]
    , n_bins: int = 10
) -> pl.DataFrame:
    '''
    Computes the Population Stability Index of a binary model by binning the new score into n_bins.

    Parameters
    ----------
    new_score
        Either a Polars Series or a NumPy array that contains the new probabilites
    old_score
        Either a Polars Series or a NumPy array that contains the old probabilites
    n_bins
        The number of bins used in the computation. By default it is 10, which means we are using deciles
    '''
    if isinstance(new_score, np.ndarray):
        s1 = pl.Series(new_score)
    else:
        s1 = new_score
    
    if isinstance(old_score, np.ndarray):
        s2 = pl.Series(old_score)
    else:
        s2 = old_score

    qcuts = np.arange(start=1/n_bins, stop=1.0, step = 1/n_bins)
    s1_cuts = s1.qcut(quantiles=qcuts)
    s1_summary = s1_cuts.lazy().groupby(pl.col("category").cast(pl.Utf8)).agg(
        a = pl.count()
    )

    cuts = s1_cuts.get_column("break_point").unique().sort()[:len(qcuts)]

    s2_summary = s2.cut(bins = cuts, series=False).lazy().groupby(
        pl.col("category").cast(pl.Utf8)
    ).agg(
        b = pl.count()
    )
    return s1_summary.join(s2_summary, on="category").with_columns(
        a = pl.col("a")/len(s1),
        b = pl.col("b")/len(s2)
    ).with_columns(
        a_minus_b = pl.col("a") - pl.col("b"),
        ln_a_on_b = (pl.col("a")/pl.col("b")).log()
    ).with_columns(
        psi = pl.col("a_minus_b") * pl.col("ln_a_on_b")
    ).sort("category").rename({"category":"score_range"}).collect()

def l2_loss(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , sample_weights:Optional[np.ndarray]=None
) -> float:
    '''
    Computes average L2 loss of some regression model

    Parameters
    ----------
    y_actual
        Actual target
    y_predicted
        Predicted target
    sample_weights
        An array of size (n_sample, ) which provides weights to each sample
    '''
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
    '''
    Computes average L1 loss of some regression model

    Parameters
    ----------
    y_actual
        Actual target
    y_predicted
        Predicted target
    sample_weights
        An array of size (n_sample, ) which provides weights to each sample
    '''
    diff = np.abs(y_actual - y_predicted)
    if sample_weights is None:
        return np.mean(diff)
    else:
        return (sample_weights/(len(diff))).dot(diff)

mae = l1_loss 

def r2(y_actual:np.ndarray, y_predicted:np.ndarray) -> float:
    '''
    Computes R square metric for some regression model

    Parameters
    ----------
    y_actual
        Actual target
    y_predicted
        Predicted target
    '''

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
    '''
    Computes adjusted R square metric for some regression model

    Parameters
    ----------
    y_actual
        Actual target
    y_predicted
        Predicted target
    p
        Number of predictive variables used
    '''
    df_tot = len(y_actual) - 1
    return 1 - (1 - r2(y_actual, y_predicted)) * df_tot / (df_tot - p)

def huber_loss(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , delta:float
    , sample_weights:Optional[np.ndarray]=None  
) -> float:
    '''
    Computes huber loss of some regression model

    See: https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_actual
        Actual target
    y_predicted
        Predicted target
    delta
        The delta parameter in huber loss
    sample_weights
        An array of size (n_sample, ) which provides weights to each sample
    '''
    
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
