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

# from dsds.prescreen import type_checker
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

def _tp_fp_frame(
    y_actual:Union[np.ndarray, pl.Series],
    y_predicted:Union[np.ndarray, pl.Series],
    ratio: bool = True
) -> pl.LazyFrame:
    '''
    Get true positive and false positive counts at various thresholds. The thresholds are determined
    by the probabilties the model gives. Returns a lazy dataframe with stats needed for precision,
    recall, and roc_auc calculation. If ratio is true, true positive rate and false positive rate will
    be returned instead. This is meant to be only used internally.

    Parameters
    ----------
    y_actual
        Actual binary labels
    y_predicted
        Predicted probabilities
    ratio
        Whether to return tpr, fpr instead of tp, fp
    '''
    df = pl.from_records((y_predicted, y_actual), schema=["threshold", "actual"])
    all_positives = pl.lit(y_actual.sum())
    n = len(df)
    temp = df.lazy().group_by("threshold").agg(
        pl.count().alias("cnt")
        , pl.col("actual").sum().alias("pos_cnt_at_threshold")
    ).sort("threshold").with_columns(
        (pl.lit(n) - pl.col("cnt").cumsum() + pl.col("cnt")).alias("predicted_positive")
        , (
            all_positives - pl.col("pos_cnt_at_threshold").cumsum()
        ).shift_and_fill(fill_value=all_positives, periods=1).alias("tp")
    ).select(
        pl.col("threshold")
        , pl.col("cnt")
        , pl.col("pos_cnt_at_threshold")
        , pl.col("tp")
        , (pl.col("predicted_positive") - pl.col("tp")).alias("fp")
        , (pl.col("tp") / pl.col("predicted_positive")).alias("precision")
    )
    if ratio:
        return temp.select(
            pl.col("threshold")
            , pl.col("cnt")
            , pl.col("pos_cnt_at_threshold")
            , (pl.col("tp") / pl.col("tp").first()).alias("tpr")
            , (pl.col("fp") / pl.col("fp").first()).alias("fpr")
            , pl.col("precision")
        )
    return temp

def precision_recall(
    y_actual:Union[np.ndarray, pl.Series]
    , y_predicted:Union[np.ndarray, pl.Series]
    , beta:Union[float, list[float]] = 1.
    , around: Optional[float] = None
) -> pl.DataFrame:
    '''
    Get precision and recall from various thresholds. Thresholds are decided by y_predicted's probabilities.
    In the multiclass case, y_actual and y_predicted should be 2D NumPy arrays.

    Parameters
    ----------
    y_actual
        Actual binary labels
    y_predicted
        Predicted probabilities
    beta
        The beta values in F_beta score. You can pass in one beta, or a list of betas.
    around
        If given, will return thresholds only around the given value (+- 0.02).
    '''
    series_input:bool = isinstance(y_actual, pl.Series) & isinstance(y_predicted, pl.Series)
    is_1d_numpy:bool = False
    if not series_input:
        is_1d_numpy = (y_actual.ndim == 1) & (y_predicted.ndim == 1)

    if isinstance(beta, float):
        f_list = [beta]
    else:
        f_list = [b for b in beta if b > 0.]

    exprs_for_f = [
        (
            pl.lit(b**2 + 1) 
            *
            (pl.col("precision") * pl.col("tpr") / (pl.lit(b**2) * pl.col("precision") + pl.col("tpr")))
        ).alias(f"F_{b:.2f}")
        for b in f_list
    ]

    if is_1d_numpy | series_input:
        out_frame = _tp_fp_frame(y_actual, y_predicted, ratio=True).select(
            pl.col("threshold")
            , pl.col("cnt").alias("predicted_cnt_at_threshold")
            , pl.col("pos_cnt_at_threshold")
            , pl.col("tpr").alias("recall")
            , pl.col("precision")
            , *exprs_for_f
        )
    else:
        # multi-class, must be NumPy Input
        if y_actual.shape[1] != y_predicted.shape[1]:
            raise ValueError("Input shapes must agree for multiclass roc auc. Found "
                             f"actual has shape {y_actual.shape} and predicted has shape {y_predicted.shape}.")
        
        frames = (
            _tp_fp_frame(y_actual[:, i].ravel(), y_predicted[:, i].ravel()).select(
                pl.lit(i).alias("class_no")
                , pl.col("threshold")
                , pl.col("cnt").alias("predicted_cnt_at_threshold")
                , pl.col("pos_cnt_at_threshold")
                , pl.col("tpr").alias("recall")
                , pl.col("precision")
                , *exprs_for_f
            )
            for i in range(y_actual.shape[1])
        )

        out_frame = pl.concat(pl.collect_all(frames))

    if around is None:
        return out_frame.lazy().collect()
    else:
        return out_frame.lazy().filter(
            pl.col("threshold").is_between(pl.lit(around - 0.02), pl.lit(around + 0.02))
        ).collect()

def roc_auc(
    y_actual:Union[np.ndarray, pl.Series],
    y_predicted:Union[np.ndarray, pl.Series],
    strategy:WeightStrategy="balanced"
) -> float:
    '''
    Return the Area Under the Curve metric for the model's predictions. For multiclass classification,
    this currently only supports aggregated roc auc for each class. Note that in the multiclass case,
    y_actual should have only one 1 per row. In the multiclass case, y_actual and y_predicted
    should be NumPy arrays.

    Parameters
    ----------
    y_actual
        Actual binary labels. Should always be array of integers.
    y_predicted
        Predicted probabilities
    strategy
        Weight strategy for multiclass roc auc. If none, the averange of each individual binary
        roc auc will be used. If balanced, the score will be balanced according to class counts.
        Custom is not supported at this moment and will be treated as none.

    Performance
    -----------
    For small arrays, length ~ 1000, Scikit-learn's implementation is faster. But for bigger ones, 
    length > 10k, this has much better performance. If it is multiclass, this is almost always 
    faster. Please measure on your own device for most accurate information. In this case, performance 
    only matters when you are repeatedly applying this function.
    ''' 
    series_input:bool = isinstance(y_actual, pl.Series) & isinstance(y_predicted, pl.Series)
    is_1d_numpy:bool = False
    if not series_input:
        is_1d_numpy = (y_actual.ndim == 1) & (y_predicted.ndim == 1)

    if is_1d_numpy | series_input:
        frame = _tp_fp_frame(y_actual, y_predicted, ratio=True).collect()
        return float(-np.trapz(frame["tpr"], frame["fpr"]))
    elif (y_actual.ndim == 2) & (y_predicted.ndim == 2):
        # Has to be not 1d numpy
        if y_actual.shape[1] != y_predicted.shape[1]:
            raise ValueError("Input shapes must agree for multiclass roc auc. Found "
                             f"actual has shape {y_actual.shape} and predicted has shape {y_predicted.shape}.")
        
        frames = (
            _tp_fp_frame(y_actual[:, i].ravel(), y_predicted[:, i].ravel())
            for i in range(y_actual.shape[1])
        )
        roc_aucs = np.array([
            -np.trapz(f["tpr"], f["fpr"]) for f in pl.collect_all(frames)
        ])
        if strategy == "balanced":
            class_count:np.ndarray = np.sum(y_actual, axis = 0) 
            class_weights:np.ndarray = class_count / np.sum(class_count)
            return class_weights.dot(roc_aucs)
        else:
            return np.mean(roc_aucs)
    else:
        raise ValueError("Input shapes must be either both 1 dim or both 2 dim. Found "
                        f"actual has shape {y_actual.shape} and predicted has shape {y_predicted.shape}.")

def logloss(
    y_actual:Union[np.ndarray, pl.Series]
    , y_predicted:Union[np.ndarray, pl.Series]
    , sample_weights:Optional[np.ndarray]=None
    # , min_prob:float = 1e-12
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
    min_prob : Removed. Unless requested by people.
        Minimum probability to clip so that we can prevent illegal computations like 
        log(0). If p < min_prob, log(min_prob) will be computed instead.
    '''
    # Takes about 1/3 time of sklearn's log_loss because we parallelized some computations
    y_a, y_p = _flatten_input(y_actual, y_predicted)
    if check_binary:
        uniques = np.unique(y_a)
        if uniques.size != 2:
            raise ValueError("Currently this only supports binary classification.")
        if not (0 in uniques) & (1 in uniques):
            raise ValueError("Currently this only supports binary classification with 0 and 1 target.")

    if sample_weights is None:
        return pl.from_records((y_a, y_p), schema=["y", "p"]).with_columns(
            pl.col("p").log().alias("l"),
            ((pl.lit(1., dtype=pl.Float64)-pl.col("p")).log()).alias("o"),
            (pl.lit(1., dtype=pl.Float64) - pl.col("y")).alias("ny")
        ).select(
            pl.sum_horizontal(
                pl.col("y").dot(pl.col("l")),
                pl.col("ny").dot(pl.col("o"))
            ) * pl.lit((-1/len(y_a)), dtype=pl.Float64)
        ).item(0,0)
    else:
        s = sample_weights.ravel()
        return pl.from_records((y_a, y_p, s), schema=["y", "p", "s"]).with_columns(
            (pl.col("s") * pl.col("p").log()).alias("l"),
            (pl.col("s") * (pl.lit(1., dtype=pl.Float64) - pl.col("p")).log()).alias("o"),
            (pl.lit(1., dtype=pl.Float64) - pl.col("y")).alias("ny")
        ).select(
            pl.sum_horizontal(
                pl.col("y").dot(pl.col("l")),
                pl.col("ny").dot(pl.col("o"))
            ) * pl.lit((-1/len(y_a)), dtype=pl.Float64)
        ).item(0,0)
    
def psi_discrete(
    expected: pl.Series,
    actual: pl.Series,
    full_table: bool = False
) -> pl.DataFrame:
    '''
    Computes the Population Stability Index of string series.

    Parameters
    ----------
    expected
        Either a Polars Series or a NumPy array that contains the expected values
    actual
        Either a Polars Series or a NumPy array that contains the actual values
    full_table
        If true, will return the full table used in PSI computation and you can see which bin contributes
        the most for the change. If false, a 1x1 dataframe will be returned with only the total PSI. If you
        want a floating point result, do psi(new, old, n_bins, False).item(0,0)
    '''
    df1 = expected.value_counts(parallel=True).lazy()
    df2 = actual.value_counts(parallel=True).lazy()
    table = (
        df1.join(df2, left_on=expected.name, right_on=actual.name, how="outer", suffix="_right")
        .select(
            pl.col(expected.name),
            (pl.col("counts") / len(expected)).clip_min(0.00001).alias("e"),
            (pl.col("counts_right") / len(actual)).clip_min(0.00001).alias("a")
        ).with_columns(
            (pl.col("e") - pl.col("a")).alias(r"e% - a%"),
            (pl.col("e")/pl.col("a")).log().alias("ln_e_on_a")
        ).with_columns(
            (pl.col(r"e% - a%") * pl.col("ln_e_on_a")).alias("psi")
        )
    )
    if full_table:
        return table.sort(expected.name).collect()
    else:
        return table.select(pl.col("psi").sum().alias("psi")).collect()
    
def psi(
    expected: Union[pl.Series, np.ndarray]
    , actual: Union[pl.Series, np.ndarray]
    , n_bins: int = 10
    , full_table: bool = False
) -> pl.DataFrame:
    '''
    Computes the Population Stability Index of a new continuous variable vs. an old continuous variable by
    binning new series into n_bins using quantiles.

    Parameters
    ----------
    expected
        Either a Polars Series or a NumPy array that contains the new probabilites
    actual
        Either a Polars Series or a NumPy array that contains the old probabilites
    n_bins
        The number of bins used in the computation. By default it is 10, which means we are using deciles
    full_table
        If true, will return the full table used in PSI computation and you can see which bin contributes
        the most for the change. If false, a 1x1 dataframe will be returned with only the total PSI. If you
        want a floating point result, do psi(new, old, n_bins, False).item(0,0)
    '''
    s1 = pl.Series(expected)
    s2 = pl.Series(actual)

    s1_cuts:pl.DataFrame = s1.qcut(n_bins, include_breaks=True).struct.unnest()\
        .filter(pl.col("break_point").is_not_null())

    s1_summary = s1_cuts.lazy().group_by(pl.col("category").cast(pl.Utf8)).agg(
        a = pl.count(),
        break_point = pl.col("break_point").first()
    ).collect()
    s2_base:pl.DataFrame = s2.cut(breaks = s1_summary.filter(pl.col("break_point").is_finite())["break_point"].sort(), 
                                    include_breaks=True).struct.unnest()

    s2_summary:pl.DataFrame = s2_base.lazy().group_by(
        pl.col("category").cast(pl.Utf8)
    ).agg(
        b = pl.count()
    )
    table = s1_summary.lazy().join(s2_summary, on="category").with_columns(
        (pl.col("a")/len(s1)).clip_min(0.00001).alias("e"),
        (pl.col("b")/len(s2)).clip_min(0.00001).alias("a")
    ).with_columns(
        (pl.col("e") - pl.col("a")).alias(r"e% - a%"),
        (pl.col("e")/pl.col("a")).log().alias("ln_e_on_a")
    ).with_columns(
        (pl.col(r"e% - a%") * pl.col("ln_e_on_a")).alias("psi")
    )
    if full_table:
        return table.sort("category").rename({"category":"range"})\
                .select("range", "break_point"
                        , pl.col("e").alias("expected%")
                        , pl.col("a").alias("actual%")
                        , r"e% - a%", "psi").collect()
    else:
        return table.select(pl.col("psi").sum().alias("psi")).collect()

def mse(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
    , sample_weights:Optional[np.ndarray]=None
) -> float:
    '''
    Computes average mean square error of some regression model. Currently only supports 1d arrays.

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

def rmsle(
    y_actual:np.ndarray
    , y_predicted:np.ndarray
) -> float:
    ''' 
    Computes the Root Mean Squared Logarithmic Error.

    Parameters
    ----------
    y_actual
        Actual target
    y_predicted
        Predicted target
    '''
    
    df = pl.from_records([y_actual, y_predicted], schema=["a", "p"]).with_columns(
        (pl.col("a") + pl.lit(1.)).alias("a1"),
        (pl.col("p") + pl.lit(1.)).alias("p1")
    ).select(
        (pl.col("p1")/pl.col("a1")).log().pow(2).mean()
    )

    return df.item(0,0)

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

    Performance: if rows in x, y are normalized, then you may set normalize to False and this will
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
    if (y is None) | (x is y):
        return rs_self_cosine_similarity(x, normalize)
    elif (x.ndim == 1) & (y.ndim == 1):
        if normalize:
            return x.dot(y)/np.sqrt(x.dot(x) * y.dot(y))
        return x.dot(y)
    else:
        return rs_cosine_similarity(x, y, normalize)
    
def cosine_dist(x:np.ndarray, y:Optional[np.ndarray]=None) -> np.ndarray:
    return 1 - cosine_similarity(x,y,True)

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
        when the lists are lists of words. All stopwords will also be removed before counting. Set this to False
        if stemming doesn't matter or if you want better performance.
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
    Computes pairwise jaccard similarity between two list columns.

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
    # _ = type_checker(df, [c1,c2], "list", "df_jaccard_similarity")
    out:pl.DataFrame = rs_df_inner_list_jaccard(df, c1, c2, inner_dtype, include_null)
    if append:
        return pl.concat([df, out], how="horizontal")
    return out