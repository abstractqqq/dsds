from .prescreen import (
    infer_discretes
    , check_binary_target
    , check_binary_target_col
    , get_numeric_cols
    , get_unique_count
    , get_string_cols
    , type_checker
)

from .type_alias import (
    PolarsFrame
    , MRMRStrategy
    , BinaryModels
    , clean_strategy_str
    , ClassifModel
)
from .blueprint import(
    _dsds_select
)
from .sample import (
    train_test_split
)
from .metrics import (
    logloss
    , roc_auc
)
from typing import (
    Any,
    Optional, 
    Tuple, 
    Union,
)
from itertools import combinations
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.stats import ks_2samp
from scipy.special import fdtrc, psi
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import polars as pl
import numpy as np
import math
import dsds

logger = logging.getLogger(__name__)

def abs_corr(
    df: PolarsFrame
    , target: str
    , cols: Optional[list[str]] = None
) -> pl.DataFrame:
    '''
    Returns a dataframe with features and their |correlation| with target. NaN correlation
    will be filled with -999. Note this makes sense since NaN is caused by 0 variance 
    (constant data) in most situations.

    Parameters
    ----------
    df 
        Either an eager or lazy Polars dataframe
    target
        The target column
    cols
        List of numerical columns. If not provided, will use all numerical columns
    '''
    if isinstance(cols, list):
        _ = type_checker(df, cols, "numeric", "corr_filter")
        nums = cols
    else:
        nums = get_numeric_cols(df)

    return (
        df.lazy().select(pl.corr(c, target).abs() for c in nums)
        .fill_nan(pl.lit(-999.0))
        .collect()
        .transpose(include_header=True, column_names=["abs_corr"])
        .sort("abs_corr", descending=True)
        .set_sorted("abs_corr")
    )

def abs_corr_selector(
    df: PolarsFrame
    , target: str
    , threshold: float
) -> PolarsFrame:
    '''
    Keeps only the columns that have |correlation with target| > threshold and the ones that cannot be 
    processed by the algorithm.

    Parameters
    ----------
    df
        Either an eager or a lazy Polars DataFrame.
    target
        The target column
    threshold
        The threshold above which the features will be selected
    '''
    nums = get_numeric_cols(df, exclude=[target])
    complement = [f for f in df.columns if f not in nums]
    # select high corr columns
    to_select = abs_corr(df, target, nums)\
                .filter(pl.col("abs_corr") >= threshold)["column"].to_list()
    print(f"Selected {len(to_select)} features. There are {len(complement)} columns the algorithm "
          "cannot process. They are also returned.")
    # add the complement set
    return _dsds_select(df, to_select + complement)

def discrete_ig(
    df:pl.DataFrame
    , target:str
    , cols:Optional[list[str]] = None
) -> pl.DataFrame:

    if isinstance(cols, list):
        discretes = cols
    else: # If discrete_cols is not passed, infer it.
        discretes = infer_discretes(df, exclude=[target])

    # Compute target entropy. This only needs to be done once.
    target_entropy = df.group_by(target).agg(
                        (pl.count()).alias("prob(target)") / len(df)
                    )["prob(target)"].entropy()

    # Get unique count for selected columns. This is because higher unique percentage may skew information gain
    unique_count = get_unique_count(df.select(discretes)).with_columns(
        (pl.col("n_unique") / len(df)).alias("unique_pct")
    ).rename({"column":"feature"})

    conditional_entropy = (
        df.lazy().group_by(target, pred).agg(
            pl.count()
        ).with_columns(
            (pl.col("count").sum().over(pred) / len(df)).alias("prob(predictive)"),
            (pl.col("count") / pl.col("count").sum()).alias("prob(target,predictive)")
        ).select(
            pl.lit(pred, dtype=pl.Utf8).alias("feature"),
            (-((pl.col("prob(target,predictive)")/pl.col("prob(predictive)")).log() 
            * pl.col("prob(target,predictive)")).sum()).alias("conditional_entropy") 
        )
        for pred in discretes
    )

    return pl.concat(pl.collect_all(conditional_entropy))\
        .with_columns(
            target_entropy = pl.lit(target_entropy),
            information_gain = pl.max_horizontal(pl.lit(target_entropy) - pl.col("conditional_entropy"), 0)
        ).join(unique_count, on="feature")\
        .select("feature", "target_entropy", "conditional_entropy", "unique_pct", "information_gain")\
        .with_columns(
            weighted_information_gain = (1 - pl.col("unique_pct")) * pl.col("information_gain")
        )

discrete_mi = discrete_ig

def discrete_ig_selector(
    df:PolarsFrame
    , target:str
    , top_k:int
) -> PolarsFrame:
    '''
    Keeps only the top_k features in terms of discrete_ig and the ones that cannot be processed by the algorithm.

    Parameters
    ----------
    df
        Either an eager or lazy dataframe. If lazy, it will be collected
    target
        The target column
    top_k
        Only the top_k features in terms of discrete_ig will be selected 
    '''

    input_data:pl.DataFrame = df.lazy().collect()
    discrete_cols = infer_discretes(df, exclude=[target])
    complement = [f for f in df.columns if f not in discrete_cols]
    to_select = discrete_ig(input_data, target, discrete_cols)\
        .top_k(by="information_gain", k = top_k)["feature"].to_list()

    print(f"Selected {len(to_select)} features. There are {len(complement)} columns the "
          "algorithm cannot process. They are also returned.")

    return _dsds_select(df, to_select + complement)

def mutual_info(
    df:pl.DataFrame
    , target:str
    , conti_cols:list[str]
    , n_neighbors:int=3
    , seed:int=42
) -> pl.DataFrame:
    '''
    Approximates mutual information (information gain) between the continuous variables and the target. This
    is essentially the same as sklearn's implementation, except that

    1. This uses Scipy library's kdtree, instead of sklearn's kdtree and nearneighbors
    2. This uses all cores by default
    3. There are less "checks" and "safeguards", meaning input data quality is expected to be "good".
    4. Conti_cols are supposed to be "continuous" variables. In sklearn's mutual_info_classif, if you input a dense 
        matrix X, it will always be treated as continuous, and if X is sparse, it will be treated as discrete.

    Parameters
    ----------
    df
        An eager dataframe
    target
        The target column
    conti_cols
        A list of columns with continuous values
    n_neighbors
        Number of neighbors. Used in the approximation method provided by the paper
    seed
        The random seed used to generate noise, which prevents points to collide and cause difficulty for the
        nearest neighbor method used in the approximation

    Sources
    -------
        (1). B. C. Ross “Mutual Information between Discrete and Continuous Data Sets”. PLoS ONE 9(2), 2014.\n
        (2). A. Kraskov, H. Stogbauer and P. Grassberger, “Estimating mutual information”. Phys. Rev. E 69, 2004. 
    '''
    n = len(df)
    rng = np.random.default_rng(seed)
    target_col = df[target].to_numpy().ravel()
    unique_targets = np.unique(target_col)
    all_masks = {}
    for t in unique_targets:
        all_masks[t] = target_col == t
        if np.sum(all_masks[t]) <= n_neighbors:
            raise ValueError(f"The target class {t} must have more than {n_neighbors} values in the dataset.")        

    estimates = []
    psi_n_and_k = psi(n) + psi(n_neighbors)
    pbar = tqdm(total = len(conti_cols), desc = "Mutual Info")
    for col in df.select(conti_cols).get_columns():
        if col.null_count() > 0:
            logger.warn(f"Found column {col.name} has null values. It is filled with the mean of the column. "
                        "It is highly recommended that you impute the column beforehand.")
            c = col.fill_null(col.mean()).cast(pl.Float64).to_numpy().reshape(-1,1)
        else:
            c = col.cast(pl.Float64).to_numpy().reshape(-1,1)
        # Add random noise here because if inpute data is too big, then adding
        # a random matrix of the same size will require a lot of memory upfront.
        c = c + (1e-10 * np.mean(c) * rng.standard_normal(size=c.shape)) 
        radius = np.empty(n)
        label_counts = np.empty(n)
        for t in unique_targets:
            mask = all_masks[t]
            c_masked = c[mask]
            kd1 = KDTree(data=c_masked, leafsize=40)
            # dd = distances from the points the the k nearest points. +1 because this starts from 0. It is 1 off from 
            # sklearn's kdtree.
            dd, _ = kd1.query(c_masked, k = n_neighbors + 1, workers=dsds.THREADS)
            radius[mask] = np.nextafter(dd[:, -1], 0)
            label_counts[mask] = np.sum(mask)

        kd2 = KDTree(data=c, leafsize=40) 
        m_all = kd2.query_ball_point(c, r = radius, return_length=True, workers=dsds.THREADS)
        estimates.append(
            max(0, psi_n_and_k - np.mean(psi(label_counts) + psi(m_all)))
        ) # smallest is 0
        pbar.update(1)

    pbar.close()
    return pl.from_records((conti_cols, estimates), schema=["feature", "estimated_mi"])

# Selectors should always return target
def mutual_info_selector(
    df:PolarsFrame
    , target:str
    , n_neighbors:int=3
    , top_k:int = 50
    , n_threads:int= dsds.THREADS
    , seed:int=42
) -> PolarsFrame:
    '''
    Keeps only the top_k features in terms of mutual_info_score and the ones that cannot be processed by the algorithm.

    Parameters
    ----------
    df
        Either an eager or lazy Polars dataframe. If lazy, it will be collected
    target
        The target column
    n_neighbors
        The n_neighbors parameter in the approximation method
    top_k
        The top_k features will ke kept
    n_threads
        The max number of workers for multithreading
    seed
        Random seed used in approximation to generate noise
    '''
    input_data:pl.DataFrame = df.lazy().collect()
    nums = get_numeric_cols(df, exclude=[target])
    complement = [f for f in df.columns if f not in nums]
    to_select = mutual_info(input_data, target, nums, n_neighbors, seed)\
                .top_k(by="estimated_mi", k = top_k)["feature"].to_list()

    logger.info(f"Selected {len(to_select)} features. There are {len(complement)} columns the "
          "algorithm cannot process. They are also returned.")

    return _dsds_select(df, to_select + complement, persist=True)

def _f_score(
    df:PolarsFrame
    , target:str
    , num_list:list[str]
) -> np.ndarray:
    '''
    This is the same as what is in f_classif to compute f_score. Except that this only 
    returns a numpy array of f scores and this does not error check.
    '''
    
    step_one_expr:list[pl.Expr] = [pl.count().alias("cnt")] 
    step_two_expr:list[pl.Expr] = []
    for n in num_list:
        n_sum:str = n + "_sum" # sum of class
        n_var:str = n + "_var" # var within class
        step_one_expr.append(
            pl.col(n).sum().alias(n_sum)
        )
        step_one_expr.append(
            pl.col(n).var(ddof=0).alias(n_var) 
        )
        step_two_expr.append(
            (pl.col(n_sum)/pl.col("cnt") - pl.col(n_sum).sum()/pl.col("cnt").sum()).pow(2).dot(pl.col("cnt")) 
            / pl.col(n_var).dot(pl.col("cnt"))
        )

    ref = (
        df.lazy().group_by(target).agg(step_one_expr)
        .select(
            pl.col("cnt").sum().alias("n_samples")
            , pl.col(target).count().alias("n_classes")
            , *step_two_expr
        ).collect()
    )
    
    n_samples = ref.drop_in_place("n_samples")[0]
    n_classes = ref.drop_in_place("n_classes")[0]
    df_btw_class = n_classes - 1 
    df_in_class = n_samples - n_classes

    return ref.to_numpy().ravel() * (df_in_class / df_btw_class)

def f_classif(
    df:PolarsFrame
    , target:str
    , cols:Optional[list[str]]=None
) -> pl.DataFrame:
    '''
    Computes ANOVA one way test, the f value/score and the p value. Equivalent to f_classif in sklearn.feature_selection
    , but is more dataframe-friendly and faster. 

    Parameters
    ----------
    df
        Either a lazy or an eager Polars DataFrame
    target
        The target column
    cols
        If not provided, will use all inferred numeric columns
    '''
    if isinstance(cols, list):
        nums = cols
    else:
        nums = get_numeric_cols(df, exclude=[target])

    step_one_expr:list[pl.Expr] = [pl.count().alias("cnt")] 
    step_two_expr:list[pl.Expr] = []
    for n in nums:
        n_sum:str = n + "_sum" # sum of class
        n_var:str = n + "_var" # var within class
        step_one_expr.append(
            pl.col(n).sum().alias(n_sum)
        )
        step_one_expr.append(
            pl.col(n).var(ddof=0).alias(n_var) 
        )
        step_two_expr.append(
            (pl.col(n_sum)/pl.col("cnt") - pl.col(n_sum).sum()/pl.col("cnt").sum()).pow(2).dot(pl.col("cnt")) 
            / pl.col(n_var).dot(pl.col("cnt"))
        )

    ref = (
        df.lazy().group_by(target).agg(step_one_expr)
        .select(
            pl.col("cnt").sum().alias("n_samples")
            , pl.col(target).len().alias("n_classes")
            , *step_two_expr
        ).collect()
    )
    n_samples = ref.drop_in_place("n_samples")[0]
    n_classes = ref.drop_in_place("n_classes")[0]
    df_btw_class = n_classes - 1 
    df_in_class = n_samples - n_classes

    if df_btw_class == 0:
        raise ZeroDivisionError("Target has only one class.")
    
    f_values = ref.to_numpy().ravel() * (df_in_class / df_btw_class)
    # We should scale this by (df_in_class / df_btw_class) because we did not do this earlier
    # At this point, f_values should be a pretty small dataframe. 
    # Cast to numpy, so that fdtrc can process it properly.

    p_values = fdtrc(df_btw_class, df_in_class, f_values) # get p values 
    return pl.from_records((nums, f_values, p_values), schema=["feature","f_value","p_value"])

def f_score_selector(
    df:PolarsFrame
    , target:str
    , top_k:int
) -> PolarsFrame:
    '''
    Keeps only the top_k features in terms of f-score and the ones that cannot be processed by the algorithm.

    Parameters
    ----------
    df
        Either an eager or lazy Polars dataframe. If lazy, it will be collected
    target
        The target column
    top_k
        The top_k features will ke kept
    '''
    input_data:pl.DataFrame = df.lazy().collect()
    nums = get_numeric_cols(input_data, exclude=[target])
    complement = [f for f in df.columns if f not in nums]
    scores = _f_score(input_data, target, nums)
    to_select = pl.DataFrame({"feature":nums, "fscore":scores})\
        .top_k(by = "fscore", k = top_k)\
        .get_column("feature").to_list()

    print(f"Selected {len(to_select)} features. There are {len(complement)} columns the "
          "algorithm cannot process. They are also returned.")

    return _dsds_select(df, to_select + complement, persist=True)

def _ks_2_samp(
    feature: np.ndarray
    , target: np.ndarray
    , i: int
) -> Tuple[float, float, int]:
    ''' 
    Computes the ks-statistics for the feature on class 0 and class 1. The bigger the ks
    statistic, that means the feature has greater differences on each class. This
    function will return (ks-statistic, p-value, i). Nulls will be dropped during the 
    computation.

    Parameters
    ----------
    feature
        Feature column. Either numpy array or polars series
    target
        Target column. Either numpy array of polars series
    i
        A passthrough of the index of the feature. Not used. Only used to keep
        track of indices when this is being called in a multithreaded context.

    '''
    # if check_binary & (not check_binary_target_col(target)):
    #     raise ValueError("Target is not properly binary.")

    # Drop nulls as they will cause problems for ks computation
    valid = ~np.isnan(feature)
    use_feature = feature[valid]
    use_target = target[valid]
    # Start computing
    class_0 = (use_target == 0)
    feature_0 = use_feature[class_0]
    feature_1 = use_feature[~class_0]
    res = ks_2samp(feature_1, feature_0)
    return (res.statistic, res.pvalue, i)

def ks_statistic(
    df: pl.DataFrame
    , target: str
    , cols: Optional[list[str]]=None
) -> pl.DataFrame:
    ''' 
    Computes the ks-statistics for the feature on class 0 and class 1. The bigger the ks
    statistic for the feature, the greater differences the feature shows on each class. Nulls
    will be dropped during the computation.

    Parameters
    ----------
    df
        An eager Polars dataframe
    target
        Name of target column
    cols
        If not provided, will use all inferred numeric columns
    '''
    if cols is None:
        nums = get_numeric_cols(df, exclude=[target])
    else:
        _ = type_checker(df, nums, "numeric", "ks_statistic")
        nums = [c for c in cols if c != target]

    target_col = df[target].to_numpy(zero_copy_only=True)
    if not check_binary_target_col(target_col):
        raise ValueError("KS statistic only works when target is binary.")

    ks_values = np.zeros(shape=len(nums))
    p_values = np.zeros(shape=len(nums))
    pbar = tqdm(total=len(nums), desc="KS", position=0, leave=True)
    with ThreadPoolExecutor(max_workers=dsds.THREADS) as ex:
        futures = (
            ex.submit(_ks_2_samp, df[c].to_numpy(), target_col, i)
            for i, c in enumerate(nums)
        )
        for f in as_completed(futures):
            ks, p, i = f.result()
            ks_values[i] = ks
            p_values[i] = p
            pbar.update(1)
    
    pbar.close()
    return pl.from_records([nums, ks_values, p_values], schema=["feature", "ks", "p_value"])

def _mrmr_relevance(
    df:pl.DataFrame
    , target:str
    , cols:list[str]
    , strategy:MRMRStrategy
    , params:dict[str,Any]
) -> np.ndarray:
    
    logger.info(f"Running {strategy} to determine feature relevance...")
    s = clean_strategy_str(strategy)
    if s in ("fscore", "f", "f_score"):
        scores = _f_score(df, target, cols)
    elif s in ("mis", "mutual_info_score"):
        scores = mutual_info(df, conti_cols=cols, target=target).get_column("estimated_mi").to_numpy().ravel()
    elif s in ("lgbm", "lightgbm"):
        from lightgbm import LGBMClassifier
        print("LightGBM is not deterministic by default. Results may vary.")
        lgbm = LGBMClassifier(**params)
        lgbm.fit(df[cols].to_numpy(), df[target].to_numpy().ravel())
        scores = lgbm.feature_importances_
    else: # Pythonic nonsense
        raise ValueError(f"The strategy {strategy} is not a valid MRMR Strategy.")
    
    invalid = np.isinf(scores) | np.isnan(scores)
    if invalid.any():
        invalid_cols = [cols[i] for i, v in enumerate(invalid) if v]
        logger.info(f"Found Inf/NaN in relevance score computation. {invalid_cols}")
        logger.info("They will be set to 0. The cause is usually high null, or low variance, or "
                    "the algorithm chosen cannot handle the input data type.")        
        scores[invalid] = 0.

    return scores

# Add an option for a score the user can pass in?
def mrmr(
    df:pl.DataFrame
    , target:str
    , k:int
    , cols:Optional[list[str]] = None
    , strategy: MRMRStrategy = "fscore"
    , params:Optional[dict[str,Any]] = None
    , return_score:bool=False
) -> Union[list[str], Tuple[list[str], np.ndarray]]:
    '''
    Implements MRMR. First we have to use a strategy to find the "relevance" of a feature, and then 
    we use accumulated correlation as a criterion to select the featuers. First, we pick the top feature
    with highest "relevance". When we pick the second feature, we look at the candidate feature's abs correlation
    with the picked feature, rescale "relevance" by abs correlation, and select the second feature based on
    this rescaled "relevance". When we pick the third feature, we look at the candidate feature's accumulated
    abs correlation with the two features selected, rescale "relevance" by the accumulated abs correlation, and select
    the next most relevant feature, and the process continues until we've selected k features.
    
    Note: A common source of numerical `error` is data quality. If input data has too high null% or too 
    low variance, some methods will not work.

    Currently this only supports binary classification. It is possible to make it run even faster if we allow
    the computation of correlation matrix upfront. But in practice that often creates memory problems and therefore
    that option is removed.

    Parameters
    ----------
    df
        Either a lazy or eager Polars Dataframe. This method will collect internally.
    target
        Target column
    k
        Top k features to keep
    cols
        Optional. A list of numerical columns. If not provided, all numerical columns will be used.
    strategy
        MRMR strategy. By default, `fscore` will be used.
    params
        Optional. If a model strategy is selected (`rf`, `xgb`, `lgbm`), params is a dict of 
        parameters for the model.
    return_score
        If true, the relevance score will be returned as well
    '''
    if isinstance(cols, list):
        nums = [c for c in cols if c !=target and c in df.columns]
        _ = type_checker(df, nums, "numeric", "mrmr")
    else:
        nums = get_numeric_cols(df, exclude=[target])

    # Unlike knockout MRMR, here we should proceed with eager
    df_local = df.lazy().select(nums + [target]).collect()
    scores = _mrmr_relevance(
        df_local
        , target
        , nums
        , strategy
        , params = {} if params is None else params
    )

    # Always do low memory option. Do not scale upfront.

    # Init MRMR
    output_size = min(k, len(nums))
    logger.info(f"Found {len(nums)} total features to select from. Proceeding to select top {output_size} features.")
    # 
    acc_abs_corr = np.zeros(len(nums)) # For each feature at index i, we keep an accumulating abs corr
    top_idx = np.argmax(scores)
    selected = [nums[top_idx]]
    pbar = tqdm(total=output_size, desc = f"MRMR, {strategy}", position=0, leave=True)
    pbar.update(1)
    for j in range(1, output_size):
        argmax = -1
        current_max = -1
        last = selected[-1]
        last_selected_col: pl.Series = (df_local[last] - df_local[last].mean())
        last_selected_std: float = df_local[last].std()

        for i, candidate in enumerate(nums):
            if candidate not in selected:
                candidate_col = df_local[candidate] 
                candidate_std = candidate_col.std()
                candidate_col -= candidate_col.mean()
                a = (last_selected_col.dot(candidate_col)) / (len(df_local) * candidate_std * last_selected_std)
                # In the rare case this calculation yields a NaN, we punish by adding 1.
                # Otherwise, proceed as usual. +1 is a punishment because
                # |corr| can be at most 1. So we are enlarging the denominator, thus reducing the score.
                acc_abs_corr[i] += 1 if (np.isnan(a) | np.isinf(a)) else np.abs(a)
                denominator = acc_abs_corr[i]/j 
                new_score = scores[i] / denominator
                if new_score > current_max:
                    current_max = new_score
                    argmax = i

        selected.append(nums[argmax])
        pbar.update(1)
    pbar.close()
    print("Output is sorted in order of selection (max relevance min redundancy).")
    if return_score:
        return selected, scores 
    return selected

def mrmr_selector(
    df:PolarsFrame
    , target:str
    , top_k:int
    , strategy:MRMRStrategy = "fscore"
    , params:Optional[dict[str,Any]] = None
) -> PolarsFrame:
    '''
    Keeps only the top_k (first k) features selected by MRMR and the ones that cannot be processed by the algorithm.

    Parameters
    ----------
    df
        Either an eager or lazy Polars dataframe. If lazy, it will be collected
    target
        The target column
    top_k
        The top_k features will ke kept
    strategy
        One of 'f', 'mis', 'lgbm'. It will use the corresponding method to compute feature relevance
    params
        If any modeled relevance is used, e.g. 'rf', 'lgbm' or 'xgb', then this will be the param dict for the model
    low_memory
        If true, use less memory. But the computation will take longer
    '''
    nums = get_numeric_cols(df, exclude=[target])
    s = clean_strategy_str(strategy)
    to_select = mrmr(df, target, top_k, nums, s, params)
    logger.info(f"Selected {len(to_select)} features. There are {len(df.columns) - len(to_select)} columns the "
          "algorithm cannot process. They are also returned.")
    to_select.extend(f for f in df.columns if f not in nums)
    return _dsds_select(df, to_select, persist=True)

def knock_out_mrmr(
    df:PolarsFrame
    , target:str
    , k:int 
    , cols:Optional[list[str]] = None
    , corr_threshold:float = 0.7
    , strategy:MRMRStrategy = "fscore"
    , params:Optional[dict[str,Any]] = None
) -> list[str]:
    '''
    Essentially the same as vanilla MRMR. Instead of using avg(abs(corr)) to "weigh" punish correlated 
    variables, here we use a simpler knock out rule based on absolute correlation. We go down the list
    according to importance, take top one, knock out all other features that are highly correlated with
    it, take the next top feature that has not been knocked out, continue, until we pick enough features
    or there is no feature left. This is inspired by the package Featurewiz and its creator.

    Note that this may not guarantee to return k features when most of them are highly correlated.

    It is possible to make it run even faster if we allow the computation of correlation matrix upfront. 
    But in practice that often creates memory problems and therefore that option is removed.

    Parameters
    ----------
    df
        Either a lazy or eager Polars Dataframe. This method will collect internally.
    target
        The target column
    k
        The top k features to return
    cols
        Numerical columns to select from. If not provided, all numeric columns will be used
    corr_threshold
        The correlation threshold above which is considered too high. This means if A has high 
        correlation with B, then B will not be selected if A is already selected
    strategy
        One of 'f', 'mis', 'lgbm'. It will use the corresponding method to compute feature relevance
    params
        If any modeled relevance is used, e.g. 'lgbm', then this will be the param dict for the model
    '''
    if isinstance(cols, list):
        nums = [c for c in cols if c !=target and c in df.columns]
        _ = type_checker(df, nums, "numeric", "knock_out_mrmr")
    else:
        nums = get_numeric_cols(df, exclude=[target])

    df_local = df.lazy().select(nums + [target]).collect()
    scores = _mrmr_relevance(
        df_local
        , target = target
        , cols = nums
        , strategy = strategy
        , params = {} if params is None else params
    )

    # Set up
    surviving_indices = np.full(shape=len(nums), fill_value=True) # an array of booleans
    scores = sorted(enumerate(scores), key=lambda x:x[1], reverse=True)
    selected = []
    count = np.int32(0)
    output_size = min(k, len(nums))
    pbar = tqdm(total=output_size, desc = f"Knock out MRMR, {strategy}", position=0, leave=True)
    # Run the knock outs
    for i, _ in scores:
        if surviving_indices[i]:
            feat = nums[i]
            selected.append(feat)
            feat_expr = pl.col(feat) - pl.col(feat).mean() 
            low_corr = df_local.select(
                (
                    (feat_expr.dot(pl.col(x) - pl.col(x).mean())).truediv(pl.count()).lt(
                        corr_threshold * pl.col(feat).std() * pl.col(x).std()
                    )
                ).alias(x)
                for x in nums
            ).to_numpy()[0, :]
            surviving_indices &= low_corr
            count += 1
            pbar.update(1)
        if count >= output_size:
            break

    pbar.close()
    if count < k:
        print(f"Found only {count}/{k} number of values because most of them are highly correlated and the knock out "
              "rule eliminates most of them.")

    print("Output is sorted in order of selection (max relevance min redundancy).")
    return selected

def knock_out_mrmr_selector(
    df:PolarsFrame
    , target:str
    , top_k:int 
    , corr_threshold:float = 0.7
    , strategy:MRMRStrategy = "fscore"
    , params:Optional[dict[str,Any]] = None
) -> PolarsFrame:
    '''
    Keeps only the top_k (first k) features selected by MRMR and the ones that cannot be processed by the algorithm.

    Parameters
    ----------
    df
        Either an eager or lazy Polars dataframe. If lazy, it will be collected
    target
        The target column
    top_k
        The top_k features will ke kept
    corr_threshold
        The threshold above which correlation is considered too high. This means if A has high correlation to B, then
        B will not be selected if A is
    strategy
        One of 'f', 'xgb', 'rf', 'mis', 'lgbm'. It will use the corresponding method to compute feature relevance
    params
        If any modeled relevance is used, e.g. 'rf', 'lgbm' or 'xgb', then this will be the param dict for the model
    '''
    nums = get_numeric_cols(df, exclude=[target])
    complement = [f for f in df.columns if f not in nums]
    s = clean_strategy_str(strategy)
    to_select = knock_out_mrmr(df, target, top_k, nums, s, corr_threshold, params)
    print(f"Selected {len(to_select)} features. There are {len(complement)} columns the "
          "algorithm cannot process. They are also returned.")
    
    return _dsds_select(df, to_select + complement, persist=True)

# Selectors for the methods below are not yet implemented

def woe_iv(
    df:PolarsFrame
    , target:str
    , cols:Optional[list[str]]=None
    , min_count:float = 1.
    , check_binary:bool = True
) -> pl.DataFrame:
    '''
    Computes information values for categorical variables. Notice that by using binning methods provided in 
    dsds.transform, you can turn numerical values into categorical bins.

    Parameters
    ----------
    df
        Either a lazy or eager Polars Dataframe
    target
        The target column
    cols
        If not provided, will use all string columns
    min_count
        A regularization term that prevents ln(0). This is the same as category_encoders package's 
        regularization parameter.
    check_binary
        Whether to check if target is binary or not
    '''
    if isinstance(cols, list):
        _ = type_checker(df, cols, "string", "woe_iv")
        input_cols = cols
    else:
        input_cols = get_string_cols(df)

    if check_binary:
        if not check_binary_target(df, target):
            raise ValueError("Target is not binary or not properly encoded or contains nulls.")

    results = (
        df.lazy().group_by(s).agg(
            ev = pl.col(target).sum()
            , nonev = (pl.lit(1) - pl.col(target)).sum()
        ).with_columns(
            ev_rate = (pl.col("ev") + min_count)/(pl.col("ev").sum() + 2.0*min_count)
            , nonev_rate = (pl.col("nonev") + min_count)/(pl.col("nonev").sum() + 2.0*min_count)
        ).with_columns(
            woe = (pl.col("ev_rate")/pl.col("nonev_rate")).log()
        ).select(
            pl.lit(s).alias("feature")
            , pl.col(s).alias("value")
            , pl.col("woe")
            , information_value = ((pl.col("ev_rate")-pl.col("nonev_rate")) * pl.col("woe")).sum()
        )
        for s in input_cols
    )
    return pl.concat(pl.collect_all(results))

def _binary_model_init(
    model_str:BinaryModels
    , params: dict[str, Any]
) -> ClassifModel:
    '''
    Returns a classification model. If n_job parameter is not specified, it will default to -1.

    Parameters
    ----------
    model_str
        One of 'lr', 'lgbm', 'xgb', 'rf'
    params
        The parameters for the model specified
    '''
    if "n_jobs" not in params:
        params["n_jobs"] = -1

    if model_str in ("logistic", "lr"):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**params)
    elif model_str in ("rf", "random_forest"):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
    elif model_str in ("xgb", "xgboost"):
        from xgboost import XGBClassifier
        model = XGBClassifier(**params)
    elif model_str in ("lgbm", "lightgbm"):
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**params)
    else:
        raise ValueError(f"The model {model_str} is not available.")
    
    return model

def _fc_fi(
    model_str:str
    , params:dict[str, Any]
    , target:str
    , features: Union[Tuple,list[str]]
    , train: pl.DataFrame
    , test: pl.DataFrame
)-> Tuple[Tuple[Tuple, float, float], np.ndarray]:
    '''
    Creates a classification model, evaluations model with log loss and roc_auc for each feature combination
    (fc) and feature importance (fi). It will return a tuple of the following structure: 
    ( (feature combination, log loss, roc_auc), feature_importance array) 

    Parameters
    ----------
    model_str
        One of 'lr', 'lgbm', 'xgb', 'rf'
    params
        The parameters for the model specified
    target
        The target column
    features
        Either a tuple or a list which represents the current feature combination
    train
        The training dataset. Must be eager
    test
        The testing dataset on which log loss and roc_auc will be evaluation. Must be eager
    '''
    estimator = _binary_model_init(model_str, params)
    _ = estimator.fit(train.select(features), train[target])
    y_pred = estimator.predict_proba(test.select(features))[:,1]
    y_test = test[target].to_numpy()
    fc_rec = (
        features,
        logloss(y_test, y_pred, check_binary=False),
        roc_auc(y_test, y_pred, check_binary=False)
    )
    if model_str in ("lr", "logistic"):
        fi_rec = np.abs(estimator.coef_).ravel()
    else:
        fi_rec = estimator.feature_importances_
    # fc_rec feature comb record, fi_rec feature importance record
    return fc_rec, fi_rec

def ebfs(
    df:pl.DataFrame
    , target:str
    , model_str:BinaryModels
    , params:dict[str, Any]
    , n_comb: int = 3
    , train_frac:float = 0.75
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
    Exhaustive Binary Feature Selection. 
    
    Suppose we have n features and n_comb = 2. This method will select all (n choose 2) 
    combinations of features, split dataset into a train and a test for each combination, 
    train a model on train, and compute feature importance and roc_auc and logloss, and 
    then finally put everything into two separate dataframes, the first of which will contain 
    the feature combinations and model performances, and the second will contain the min, avg, 
    max and var of feature importance of each feature in all its occurences in the training rounds.

    Notice since we split data into train and test every time for a different feature combination, the 
    average feature importance we derive naturally are `cross-validated` to a certain degree.

    This method will be extremely slow if (n choose n_comb) is a big number. All numerical columns 
    will be taken as potential features. Please encode the string columns if you want to use them
    as features here.

    If n_jobs is not provided in params, it will be defaulted to -1.

    This will return a feature combination (fc) summary and a feature importance (fi) summary. 

    Parameters
    ----------
    df
        An eager Polars DataFrame
    target
        The target column
    model_str
        one of 'lr', 'lgbm', 'xgb', 'rf'
    params
        Parameters for the model
    n_comb
        We will run this for all n choose n_comb combinations of features
    '''
    features = get_numeric_cols(df, exclude=[target])
    fi = {f:[] for f in features}
    records = []
    pbar = tqdm(total=math.comb(len(features), n_comb), desc="Combinations")
    df_keep = df.select(features + [target])
    for comb in combinations(features, r=n_comb):
        train, test = train_test_split(df_keep, train_frac)
        fc_rec, fi_rec = _fc_fi(model_str, params, target, comb, train, test) 
        records.append(fc_rec)
        for f, i in zip(fc_rec[0], fi_rec):
            fi[f].append(i)
        pbar.update(1)

    fc_summary = pl.from_records(records, schema=["combination", "logloss", "roc_auc"])
    stats = [
        (f, len(fi[f]), np.min(fi[f]), np.mean(fi[f]), np.max(fi[f]), np.std(fi[f])) for f in fi
    ]
    fi_summary = pl.from_records(stats, schema=["feature", "occurrences", "fi_min", "fi_mean", "fi_max", "fi_std"])
    pbar.close()
    return fc_summary, fi_summary

def ebfs_fc_filter(
    fc: pl.DataFrame
    , logloss_threshold:float
    , roc_auc_threshold:float
) -> list[str]:
    '''
    A filter method based on the feature combination result of ebfs.

    Parameters
    ----------
    fc
        The feature combination result from ebfs
    logloss_threshold
        The maximum logloss for the combination to be kept
    roc_auc_threshold
        The minimum roc_auc for the combination to be kept
    '''
    return fc.filter(
        (pl.col("logloss") <= logloss_threshold)
        & (pl.col("roc_auc") >= roc_auc_threshold)
    ).get_column("combination").explode().unique().to_list()

def _permute_importance(
    model:ClassifModel
    , X:pl.DataFrame
    , y: np.ndarray
    , index:int
    , k: int
) -> Tuple[float, int]:
    '''
    Computes permutation importance for a single feature.

    Parameters
    ----------
    model
        A trained classification model
    X
        An eager dataframe on which we shuffle the column at the given index and train the model
    y
        The target column turned into np.ndarray
    index
        The index of the column in X to shuffle
    k
        The number of times to repeat the shuffling
    '''
    test_score = 0.
    c = X.columns[index] # column to shuffle
    for _ in range(k):
        shuffled_df = X.with_columns(
            pl.col(c).shuffle(seed=42)
        )
        test_score += roc_auc(y, model.predict_proba(shuffled_df)[:, -1])

    return test_score, index

def permutation_importance(
    df:pl.DataFrame
    , target:str
    , model_str:BinaryModels
    , params:dict[str, Any]
    , k:int = 5
) -> pl.DataFrame:
    '''
    Computes permutation importance for every non-target column in df. Please make sure all columns are properly 
    encoded or transformed before calling this.
    
    Only works for binary classification and score = roc_auc for now.

    Parameters
    ----------
    df
        An eager Polars DataFrame
    target
        The target column
    model_str
        One of 'lr', 'lgbm', 'xgb', 'rf'
    params
        Parameters for the model
    k
        Permute the same feature k times
    '''
    features = df.columns
    features.remove(target)
    _ = type_checker(df, features, "numeric", "permutation_importance")
    estimator = _binary_model_init(model_str, params)
    estimator.fit(df[features], df[target])
    X = df[features]
    y = df[target].to_numpy()
    score = roc_auc(y, estimator.predict_proba(X)[:, -1])
    pbar = tqdm(total=len(features), desc="Analyzing Features")
    imp = np.zeros(shape=len(features))
    with ThreadPoolExecutor(max_workers=dsds.THREADS) as ex:
        futures = (
            ex.submit(
                _permute_importance,
                estimator,
                X,
                y,
                j,
                k
            )
            for j in range(len(features))
        )
        for f in as_completed(futures):
            test_score, i = f.result()
            imp[i] = score - (1/k)*test_score
            pbar.update(1)

    return pl.from_records((features, imp), schema=["feature", "permutation_importance"])