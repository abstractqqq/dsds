from typing import Tuple, Optional, Union
from .type_alias import WeightStrategy
import numpy as np 
import polars as pl
import logging

logger = logging.getLogger(__name__)

# No need to do length checking (len(y_1) == len(y_2)) because NumPy / Polars will complain for us.

def get_sample_weight(
    y_actual:np.ndarray
    , strategy:WeightStrategy="balanced"
    , weight_dict:Optional[dict[int, float]] = None
) -> np.ndarray:
    '''
    Infers sample weight from y_actual. All classes in y_actual must be "dense" categorical target variable, meaning 
    numbers starting in the list [0, ..., (n_classes - 1)], where the i th entry is the number of records in class_i.
    If a conversion from sparse target to dense target is needed, see `dsds.prescreen.sparse_to_dense_target`.

    Important: by assumption, target ranges from 0, ..., to (n_classes - 1) and each reprentative must have at least 1 
    instance. If target is encoded otherwise, unexpected results may be returned.

    Parameters
    ----------
    y_actual
        Actual labels
    strategy
        One of 'balanced', 'none', or 'custom'. If 'none', an array of ones will be returned. If 'custom', then a 
        weight_dict must be provided.
    weight_dict
        Dictionary of weights. If there are n_classes, keys must range from 0 to n_classes-1. Values will be the weights
        for the classes.
    '''
    out = np.ones(shape=y_actual.shape)
    if strategy == "none":
        return out
    elif strategy == "balanced":
        weights = len(y_actual) / (np.unique(y_actual).size * np.bincount(y_actual))
        for i, w in enumerate(weights):
            out[y_actual == i] = w
        return out
    elif strategy == "custom":
        if weight_dict is None:
            raise ValueError("If strategy == 'custom', then weight_dict must be provided.")
        if len(weight_dict) != np.unique(y_actual).size:
            raise ValueError("The input `weight_dict` must provide the weights for all class, with keys "
                    "ranging from 0 to n_classes-1.")
        
        for i in range(len(weight_dict)):
            w = weight_dict.get(i, None)
            if w is None:
                raise ValueError("The input `weight_dict` must provide the weights for all class, with keys "
                                 "ranging from 0 to n_classes-1.")
            out[y_actual == i] = w
        return out
    else:
        raise TypeError(f"Unknown weight strategy: {strategy}.")

def _flatten_input(y_actual: np.ndarray, y_predicted:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_a = y_actual.ravel()
    if y_predicted.ndim == 2:
        y_p = y_predicted[:, -1] # .ravel()
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
    # This is about 50% faster than sklearn's implementation. I know. This does not matter
    # that much, unless you are repeatedly computing roc_auc for some reasons.
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
    , check_binary:bool = False
) -> float:
    '''
    Return the logloss of the binary classification. This only works for binary target.

    Parameters
    ----------
    y_actual
        Actual binary labels
    y_predicted
        Predicted probabilities
    sample_weights
        An array of size (len(y_actual), ) which provides weights to each sample
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
            pl.lit(-1, dtype=pl.Float64) 
            * (pl.col("y").dot(pl.col("l")) + pl.col("ny").dot(pl.col("o"))) / len(y_a)
        ).item(0,0)
    else:
        s = sample_weights.ravel()
        return pl.from_records((y_a, y_p, s), schema=["y", "p", "s"]).with_columns(
            l = pl.col("s") * pl.col("p").clip_min(min_prob).log(),
            o = pl.col("s") * (1- pl.col("p")).clip_min(min_prob).log(),
            ny = 1 - pl.col("y")
        ).select(
            pl.lit(-1, dtype=pl.Float64) 
            * (pl.col("y").dot(pl.col("l")) + pl.col("ny").dot(pl.col("o"))) / len(y_a)
        ).item(0,0)
    
def binary_psi(
    new_score: Union[pl.Series, np.ndarray]
    , old_score: Union[pl.Series, np.ndarray]
    , n_bins: int = 10
) -> pl.DataFrame:
    '''
    Computes the Population Stability Index of a binary model by binning the new score into n_bins using quantiles.

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
    s1_cuts:pl.DataFrame = s1.qcut(quantiles=qcuts, series=False)
    s1_summary = s1_cuts.lazy().groupby(pl.col("category").cast(pl.Utf8)).agg(
        a = pl.count()
    )

    s2_base:pl.DataFrame = s2.cut(bins = s1_cuts.get_column("break_point").unique().sort().head(len(qcuts)), 
                                  series = False)

    s2_summary:pl.DataFrame = s2_base.lazy().groupby(
        pl.col("category").cast(pl.Utf8)
    ).agg(
        b = pl.count()
    )
    return s1_summary.join(s2_summary, on="category").with_columns(
        a = pl.max_horizontal(pl.col("a"), pl.lit(0.00001))/len(s1),
        b = pl.max_horizontal(pl.col("b"), pl.lit(0.00001))/len(s2)
    ).with_columns(
        a_minus_b = pl.col("a") - pl.col("b"),
        ln_a_on_b = (pl.col("a")/pl.col("b")).log()
    ).with_columns(
        psi = pl.col("a_minus_b") * pl.col("ln_a_on_b")
    ).sort("category").rename({"category":"score_range"}).collect()

def mse(
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
        An array of size (len(y_actual), ) which provides weights to each sample
    '''
    diff = y_actual - y_predicted
    if sample_weights is None:
        return diff.dot(diff)/len(diff)
    else:
        return (sample_weights/(len(diff))).dot(diff.dot(diff))
    
l2_loss = mse
brier_loss = mse

def mae(
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
        An array of size (len(y_actual), ) which provides weights to each sample
    '''
    diff = np.abs(y_actual - y_predicted)
    if sample_weights is None:
        return np.mean(diff)
    else:
        return (sample_weights/(len(diff))).dot(diff)

l1_loss = mae

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
        The delta parameter in huber loss. Must be positive.
    sample_weights
        An array of size (len(y_actual), ) which provides weights to each sample
    '''
    y_a = y_actual.ravel()
    y_p = y_predicted.ravel()
    
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