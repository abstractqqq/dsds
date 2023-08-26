from typing import (
    Tuple
    , Optional
    , Union
)
from .type_alias import (
    WeightStrategy
    , HashableDtypes
    # , PolarsFrame
)
from dsds._rust import (
    rs_cosine_similarity
    , rs_self_cosine_similarity
    , rs_df_inner_list_jaccard
    , rs_series_jaccard
    , rs_mse
    , rs_mae
    , rs_mape
    , rs_smape
    , rs_huber_loss
    , rs_snowball_stem_series
)

from dsds.prescreen import type_checker
import numpy as np 
import polars as pl
import logging

logger = logging.getLogger(__name__)

# Rule of thumb: all rust functions will take Array<f64> or ArrayView<f64>.
# Always do a astype on numpy arrays with copy set to False to minimize copying.

def get_sample_weight(
    y_actual:np.ndarray
    , strategy:WeightStrategy="balanced"
    , weight_dict:Optional[dict[int, float]] = None
) -> np.ndarray:
    '''
    Infers sample weight from y_actual. All classes in y_actual must be "dense" categorical target variable, meaning 
    numbers in the range [0, ..., (n_classes - 1)], where the i th entry is the number of records in class_i.
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

    Example
    -------
    >>> import dsds.metrics as me
    ... y_actual = np.array([0,0,1,1,2,2]) # balanced labels will return weights of 1
    >>> me.get_sample_weight(y_actual)
    array([1., 1., 1., 1., 1., 1.])
    >>> y_actual = np.array([0,1,1,1,2]) 
    >>> me.get_sample_weight(y_actual)
    array([1.66666667, 0.55555556, 0.55555556, 0.55555556, 1.66666667])
    '''
    out = np.ones(shape=y_actual.shape, dtype=np.float64)
    if strategy == "none":
        return out
    elif strategy == "balanced":
        weights = len(y_actual) / (np.unique(y_actual).size * np.bincount(y_actual))
        for i, w in enumerate(weights):
            out[y_actual == i] = np.float64(w)
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
                raise ValueError("The input `weight_dict` must provide the weights for all class, with integer keys "
                                 "ranging from 0 to n_classes-1.")
            out[y_actual == i] = np.float64(w)
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
    y_a, y_p = _flatten_input(y_actual, y_predicted)
    if check_binary:
        uniques = np.unique(y_a)
        if uniques.size != 2:
            raise ValueError("Currently this only supports binary classification.")
        if not (0 in uniques and 1 in uniques):
            raise ValueError("Currently this only supports binary classification with 0 and 1 target.")

    tpr, fpr, _ = get_tp_fp(y_a.astype(np.int8, copy=False), y_p, ratio=True)
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
    
def psi(
    new: Union[pl.Series, np.ndarray]
    , old: Union[pl.Series, np.ndarray]
    , n_bins: int = 10
) -> pl.DataFrame:
    '''
    Computes the Population Stability Index of a new continuous variable vs. an old continuous variable by
    binning new series into n_bins using quantiles.

    Parameters
    ----------
    new_score
        Either a Polars Series or a NumPy array that contains the new probabilites
    old_score
        Either a Polars Series or a NumPy array that contains the old probabilites
    n_bins
        The number of bins used in the computation. By default it is 10, which means we are using deciles
    '''
    s1 = pl.Series(new, dtype=pl.Float64)
    s2 = pl.Series(old, dtype=pl.Float64)

    qcuts = np.arange(start=1/n_bins, stop=1.0, step = 1/n_bins)
    s1_cuts:pl.DataFrame = s1.qcut(qcuts, series=False)
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

# def psi_cat(
#     new: PolarsFrame
#     , old: PolarsFrame
#     , col: str
# ) -> pl.DataFrame:
    
#     _ = type_checker(new, [col], "string", "psi_cat")
#     _ = type_checker(old, [col], "string", "psi_cat")
    
#     s1_summary = new.groupby(col).agg(pl.count().alias("a"))
#     s2_summary = old.groupby(col).agg(pl.count().alias("b"))

#     return s1_summary.join(s2_summary, on=col, how="left").with_columns(
#         a = pl.max_horizontal(pl.col("a"), pl.lit(0.00001))/pl.col("a").sum(),
#         b = pl.max_horizontal(pl.col("b"), pl.lit(0.00001))/pl.col("b").sum()
#     ).with_columns(
#         a_minus_b = pl.col("a") - pl.col("b"),
#         ln_a_on_b = (pl.col("a")/pl.col("b")).log()
#     ).with_columns(
#         psi = pl.col("a_minus_b") * pl.col("ln_a_on_b")
#     ).sort("category").rename({"category":"score_range"}).collect()

def mse(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , sample_weights:Optional[np.ndarray]=None
) -> float:
    '''
    Computes average mean square error of some regression model. Currently only supports 1d target.

    Parameters
    ----------
    y_actual
        Actual target
    y_predicted
        Predicted target
    sample_weights
        An array of size (len(y_actual), ) which provides weights to each sample
    '''
    return rs_mse(y_actual.astype(np.float64, copy=False), 
                  y_predicted.astype(np.float64, copy=False), 
                  sample_weights)
    
l2_loss = mse
brier_loss = mse

def rmse(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , sample_weights:Optional[np.ndarray]=None
) -> float:
    '''
    Computes RMSE of some regression model. Currently only supports 1d target.

    Parameters
    ----------
    y_actual
        Actual target
    y_predicted
        Predicted target
    sample_weights
        An array of size (len(y_actual), ) which provides weights to each sample
    '''
    return np.sqrt(mse(y_actual, y_predicted, sample_weights))

def mae(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , sample_weights:Optional[np.ndarray]=None
) -> float:
    '''
    Computes average L1 loss of some regression model. Currently only supports 1d target.

    Parameters
    ----------
    y_actual
        Actual target
    y_predicted
        Predicted target
    sample_weights
        An array of size (len(y_actual), ) which provides weights to each sample
    '''
    return rs_mae(y_actual.astype(np.float64, copy=False), 
                  y_predicted.astype(np.float64, copy=False), 
                  sample_weights)

l1_loss = mae

def mape(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , weighted: bool = False
) -> float:
    '''
    Computes the mean absolute percentage error commonly used in time series predictions.

    Paramters
    ---------
    y_actual
        Actual target
    y_predicted
        Predicted target
    weighted
        If weighted, then it is the wMAPE defined as in the wikipedia page: 
        https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    '''
    return rs_mape(y_actual.astype(np.float64, copy=False), 
                   y_predicted.astype(np.float64, copy=False), 
                   weighted)

def smape(
    y_actual:np.ndarray
    , y_predicted: np.ndarray
    , double_sum: bool = False
) -> float:
    '''
    Computes SMAPE: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Parameters
    ----------
    y_actual
        Actual target
    y_predicted
        Predicted target
    double_sum
        If true, uses the third formulation of SMAPE in the wiki. If denominator is 0, f64::MAX is returned.
        If false, uses the formulation that is always between 0% and 100%.
    '''
    return rs_smape(y_actual.astype(np.float64, copy=False), 
                   y_predicted.astype(np.float64, copy=False), 
                   double_sum)

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
    # This is trivial, and we won't really have any performance gain by using Rust or other stuff.
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
    Computes huber loss of some regression model.

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
    return rs_huber_loss(y_actual.astype(np.float64, copy=False), 
                         y_predicted.astype(np.float64, copy=False), 
                         delta, sample_weights)

def cosine_similarity(x:np.ndarray, y:Optional[np.ndarray]=None, normalize:bool=True) -> np.ndarray:
    '''
    Computes cosine similarity. If both x and y are 1-dimensional, this is the cosine similarity of two
    vectors. If y is None, this is the self cosine similarity of x. Say x has dim (N, F), representing
    N documents and F features. The self cosine similarity is the matrix where the ij-th entry represents
    the cosine similarity between doc i and doc j. When x and y are both give and > 1 dimensional, the 
    resulting matrix will have entry ij representing the cosine similarity between i-th doc in x and j-th
    doc in y.

    When both x and y are row-normalized matrices, this is equivalent to x.dot(y.t).

    Performance hint: if rows in x, y are normalized, then you may set normalize to False and this will
    greatly improve performance. Say x has dimension (m, n) and y has dimension (k, n), this method is much 
    faster than NumPy/Scikit-learn when m >> k. It is advised if m >> k, you should put x as
    the first input. The condition m >> k is quite common, when you have a large corpus x, and want to 
    compare a new entry y to the corpus. By my testing, m = 5000, n = 1000, k = 10, this is still faster. However, 
    when both m and n are large (both > 2000), NumPy Scikit-learn is faster. I am not sure why.

    Parameters
    ----------
    x
        A Numpy 1d/2d array
    y
        If none, perform cosine similarity with x and x. If provided, this will perform cosine similarity 
        between x and y.
    normalize
        If the rows of the matrices are normalized already, set this to False.
    '''
    if y is None or x is y:
        return rs_self_cosine_similarity(x, normalize)
    elif x.ndim == 1 and y.ndim == 1:
        if normalize:
            return x.dot(y)/np.sqrt(x.dot(x) * y.dot(y))
        return x.dot(y)
    else:
        return rs_cosine_similarity(x, y, normalize)
    
def cosine_dist(x:np.ndarray, y:Optional[np.ndarray]=None) -> np.ndarray:
    return 1 - cosine_similarity(x,y,True)

# def haversine_dist(
#     df:PolarsFrame
#     , x_long:str
#     , x_lat:str
#     , y_long:str
#     , y_lat:str
#     , out_name:str="haversine"
# ) -> PolarsFrame:
#     '''
    
#     '''
#     keep = df.columns
#     return df.with_columns(
#         sin_lat = ((pl.col(x_lat) - pl.col(y_lat))/2).sin().pow(2),
#         sin_long = ((pl.col(x_long) - pl.col(y_long))/2).sin().pow(2),
#         cos = pl.col(x_lat).cos() * pl.col(y_lat).cos()
#     ).select(
#         *keep,
#         pl.lit(2.0, dtype=pl.Float64)
#         * (pl.col("sin_lat") + pl.col("sin_long") * pl.col("cos")).sqrt().arcsin().alias(out_name)
#     )

def jaccard_similarity(
    s1:Union[pl.Series,list,np.ndarray]
    , s2:Union[pl.Series,list,np.ndarray]
    , expected_dtype: HashableDtypes = "string"
    , include_null:bool=False
    , stem:bool = False
    , parallel:bool=True
) -> float:
    '''
    Computes jaccard similarity between the two input list of strings or integers. Internally, both will be turned 
    into Polars Series.

    Parameters
    ----------
    s1
        The first list/series/array
    s2
        The second list/series/array
    expected_dtype
        Either "string" or "int". Dtype of s1 and s2.
    include_null
        If true, null will be counted as common. If false, they will not.
    stem
        If true and inner values are strings, then perform snowball stemming on the words. This is only useful 
        when the lists are lists of words. All stopwords will also be removed.
    parallel
        Whether to hash values in the lists in parallel. Only applies when internal data type is string.
    '''
    if expected_dtype in ("str", "string"):
        ss1 = pl.Series(s1, dtype=pl.Utf8)
        ss2 = pl.Series(s2, dtype=pl.Utf8)
    elif expected_dtype == "int":
        ss1 = pl.Series(s1, dtype=pl.Int64)
        ss2 = pl.Series(s2, dtype=pl.Int64)
    else:
        raise TypeError(f"The argument `expected_dtype` must be either string or int. Not {expected_dtype}.")
    
    if stem and expected_dtype == "string":
        ss1 = rs_snowball_stem_series(ss1, True)
        ss2 = rs_snowball_stem_series(ss2, True)

    return rs_series_jaccard(ss1, ss2, expected_dtype, include_null, parallel)

def df_jaccard_similarity(
    df: pl.DataFrame
    , c1: str
    , c2: str
    , inner_dtype:HashableDtypes
    , include_null:bool = True
    , append:bool = False
) -> pl.DataFrame:
    '''
    Computes pairwise jaccard similarity between two list columns. Currently this does not support 
    stemming for columns with list[str] values.

    Parameters
    ----------
    df
        An eager Polars dataframe
    s1
        Name of the first column
    s2
        Name of the second column
    inner_dtype
        The inner dtype of the list columns. Must be either int or string
    include_null
        If true, null/none will be counted as common. If false, they will not.
    append
        If true, the new similarity column will be appeded to df

    Example
    -------
    >>> from dsds.metrics import df_jaccard_similarity
    ... df = pl.DataFrame({
    ... "a":[["like", "hello"]]*2000
    ... , "b":[["like", "world"]]*2000
    ... })
    >>> df_jaccard_similarity(df, "a", "b", "str").head()
    shape: (5, 1)
    ┌─────────────┐
    │ a_b_jaccard │
    │ ---         │
    │ f64         │
    ╞═════════════╡
    │ 0.333333    │
    │ 0.333333    │
    │ 0.333333    │
    │ 0.333333    │
    │ 0.333333    │
    └─────────────┘
    '''
    _ = type_checker(df, [c1,c2], "list", "df_jaccard_similarity")
    out:pl.DataFrame = rs_df_inner_list_jaccard(df, c1, c2, inner_dtype, include_null)
    if append:
        return pl.concat([df, out], how="horizontal")
    return out