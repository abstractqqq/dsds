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
    , MRMRRelevance
    , MRMRSelectStrategy
    , BinaryModels
    , ClassifModel
)
from .blueprint import(
    _dsds_select
)
from .sample import (
    train_test_split
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
import dsds.metrics as me
import logging
import polars as pl
import numpy as np
import math
import dsds

logger = logging.getLogger(__name__)

# DO NOT WORK MORE ON SELECTOR
# WAIT UNTIL I DECIDE ON A FUNCTION SIGNATURE 

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

    Null values will be dropped. Please do not feed high null column into the algorithm.

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
    all_masks = {
        t: target_col == t for t in unique_targets
    }
    for t in all_masks:
        if np.sum(all_masks[t]) <= n_neighbors:
            raise ValueError(f"The target class {t} must have more than {n_neighbors} values in the dataset.")    

    estimates = []
    psi_n_and_k = psi(n) + psi(n_neighbors)
    pbar = tqdm(total = len(conti_cols), desc = "Mutual Info", disable=dsds.NO_PROGRESS_BAR)
    for col in df.select(conti_cols).get_columns():
        c = col.drop_nulls().cast(pl.Float64).to_numpy(zero_copy_only=True)
        # Add random noise here because if inpute data is too big, then adding
        # a random matrix of the same size will require a lot of memory upfront.
        c = c + (1e-12 * np.mean(c) * rng.standard_normal(size=c.shape)) 
        radius = np.empty(n)
        label_counts = np.empty(n)
        for t in unique_targets:
            mask = all_masks[t]
            c_masked = c[mask]
            kd1 = KDTree(data=c_masked, leafsize=40)
            # dd = distances from the points the the k nearest points. +1 because this starts from 0. 
            # It is 1 off from sklearn's kdtree.
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
    , seed:int=42
) -> PolarsFrame:
    '''
    Keeps only the top_k features in terms of mutual_info_score and the ones that cannot be processed 
    by the algorithm.

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

#[Rustify]
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
        Feature column.
    target
        Target column.
    i
        A passthrough of the index of the feature. Not used. Only used to keep
        track of indices when this is being called in a multithreaded context.
    '''

    # Drop nulls as they will cause problems for ks computation
    valid = ~np.isnan(feature)
    use_feature = feature[valid]
    use_target = target[valid]
    # Start computing
    class_0 = (use_target == 0)
    res = ks_2samp( use_feature[~class_0], use_feature[class_0])
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

    target_col = df[target].to_numpy()
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

# ----------------------------------------- MRMR ----------------------------------------------

def _mrmr_relevance(
    df:PolarsFrame
    , target:str
    , cols:list[str]
    , relevance:MRMRRelevance
) -> dict[str, float]:
    
    use_cols = cols if target in cols else cols + [target]
    logger.info(f"Running {relevance} to determine feature relevance...")
    if relevance == "f":
        scores = _f_score(df, target, cols)
    elif relevance == "mis":
        df_local = df.lazy().select(use_cols).collect()
        scores = (
            mutual_info(df_local, conti_cols=cols, target=target)
            .get_column("estimated_mi").to_numpy().ravel()
        )
    elif relevance == "lgbm":
        from dsds.optuna_integration import suggest_b_lgbm_hyperparams
        import lightgbm as lgb
        df_local = df.lazy().select(use_cols).collect()
        params, _ = suggest_b_lgbm_hyperparams(df_local, target, 30, "log_loss")
        params['verbosity'] = -1
        logger.info("LightGBM tuning is not deterministic by default. Results may vary.")
        logger.info(f"Recommended hyperparameters: {params}")
        X, y = df_local[cols].to_numpy(), df_local[target].to_numpy()
        data = lgb.Dataset(X, label=y)
        lgbm = lgb.train(params, data)
        scores = lgbm.feature_importance()
    elif relevance == "ks":
        df_local = df.lazy().select(use_cols).collect()
        scores = ks_statistic(df_local, target, cols).drop_in_place("ks").to_numpy()
    else: # Pythonic nonsense
        raise ValueError(f"The relevance strategy {relevance} is not a valid MRMR Strategy. "
                         "Check for typos, spelling, etc.")
    
    # Clean up
    invalid = np.isinf(scores) | np.isnan(scores)
    if invalid.any():
        invalid_cols = [cols[i] for i, v in enumerate(invalid) if v]
        logger.info(f"Found Inf/NaN in relevance score computation. {invalid_cols}")
        logger.info("They will be set to 0. The cause is usually high null, or low variance, or "
                    "the chosen algorithm cannot handle the input data type.")        
        scores[invalid] = 0.

    return dict(zip(cols, scores))

def _accum_corr_mrmr(
    df: PolarsFrame
    , k: int
    , cols: list[str]
    , scores: np.ndarray
    , k_weighted: bool = False
    , verbose: bool = False
) -> list[str]:
    '''
    Given `relevance scores`, run MRMR with accumulated absolute correlation strategy. E.g. for round j,
    there are j features already selected. We pick the j+1 -th feature by
        
        max_{x, x not selected} (score / ((sum_{f, f selected} abs_corr(x, f))/j)

    The denominator is the average abs corr of this feature with selected features. Note that denominator
    is always <= 1. So if the denominator is smaller (overall, x is less correlated with selected features),
    the relative score in this round will be bigger.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    k
        The number of features to pick
    scores
        MRMR Relevance score
    k_weighted
        If true, the correlation of x with the first selected feature will be weighted by 1, the correlation
        of x with the second selected feature will be weighted by (1 - 1/k), ..., the correlation of x with
        the j-th selected feature will be weighted by (1 - j/k)
    verbose
        If true, will print out some detail about the scores, accumulated correlation for each round.
    '''

    kw = int(k_weighted)
    output_size = min(k, len(cols))
    logger.info(f"Found {len(cols)} total features to select from. Proceeding to select top {output_size} features.")
    acc_abs_corr = np.zeros(len(cols), dtype=np.float64) # For each feature at index i, we keep an accumulating abs corr
    selected = [cols[int(np.argmax(scores))]]

    pbar = tqdm(total=output_size, desc = "MRMR", position=0, leave=True, disable=dsds.NO_PROGRESS_BAR)
    pbar.update(1)
    # Memoization, only for mean and std
    # If we memoize the scaled series, memory footprint will still be huge
    mean_std = df.lazy().select(
        pl.concat_list(pl.col(c).mean(), pl.col(c).std(ddof=0)) for c in cols
    ).collect().row(0)
    # dict[str, pl.Series]
    memo:dict[str, Tuple[float, float]] = dict(zip(cols, mean_std))
    for j in range(1, output_size):
        last = selected[-1]
        last_mean, last_std = memo[last]        
        last_col: pl.Expr = pl.col(last) - last_mean
        # Compute all abs correlation that we need
        abs_corrs = df.lazy().select(
            pl.lit(-k).alias(x)
            if x in selected else 
            last_col.dot(pl.col(x) - memo[x][0]).abs().truediv(pl.count() * memo[x][1] * last_std).alias(x)
            for x in cols # x: candidate
        ).collect().to_numpy()[0, :]
        # Punish by setting |corr| to 1 if NaN of Inf.
        bad = np.isnan(abs_corrs) | np.isinf(abs_corrs)
        abs_corrs[bad] = 1.0 
        # Add to accumulated abs correlation
        acc_abs_corr += np.multiply(1 - kw * j / k,  abs_corrs)
        # Compute the scaled score (the relative score)
        new_score = np.divide(j*scores, acc_abs_corr)
        # Selected ones will have negative values, so won't affect argmax 
        
        if verbose:
            top = sorted(zip(cols, new_score, acc_abs_corr), key = lambda x:x[1], reverse=True)[:20]
            top = [t for t in top if t[2] > 0]
            logger.info(f"Round {j+1}: The top {len(top)} features, relative score, and "
                        f"the accumulated correlation are the following:\n{top}")
            logger.info(f"The selected feature is {top[0][0]}")

        chosen_idx = int(np.argmax(new_score))
        selected.append(cols[chosen_idx])
        scores[chosen_idx] = 0.
        pbar.update(1)
    pbar.close()
    return selected

def _knock_out_mrmr(
    df: PolarsFrame
    , k: int
    , cols: list[str]
    , scores: np.ndarray
    , corr_threshold: float
    , verbose: bool = False
) -> list[str]:
    '''
    Given `relevance scores`, run MRMR with simple knock out strategy. E.g. for round j, pick
    a the top remaining feature. Find all remaining features with absolute correlation that is
    higher than corr_threshold with the selected feature. Knock them out. Then find the next 
    highest-score feature remaining, and continue. 

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    k
        The number of features to pick
    scores
        MRMR Relevance score
    corr_threshold
        The absolute correlation threshold to perform the knock out
    '''
    
    # Set up
    surviving_indices = np.full(shape=len(cols), fill_value=True) # an array of booleans
    scores = sorted(enumerate(scores), key=lambda x:x[1], reverse=True)
    selected = []
    output_size = min(k, len(cols))
    pbar = tqdm(total=output_size, desc = "Knock out MRMR", position=0, leave=True, disable=dsds.NO_PROGRESS_BAR)
    # Memoization, only for mean and std
    # If we memoize the scaled series, memory footprint will be too big
    mean_std = df.lazy().select(
        pl.concat_list(pl.col(c).mean(), pl.col(c).std(ddof=0)) for c in cols
    ).collect().row(0)
    memo:dict[str, Tuple[float, float]] = dict(zip(cols, mean_std))
    # Run the knock outs
    for i, _ in scores:
        if surviving_indices[i]:
            feat = cols[i]
            selected.append(feat)
            feat_mean, feat_std = memo[feat]
            feat_expr = pl.col(feat) - feat_mean
            low_corr = df.lazy().select(
                feat_expr.dot(pl.col(x) - memo[x][0]).abs().truediv(pl.count())
                .lt(corr_threshold * feat_std * memo[x][1]).alias(x)
                if surviving_indices[j] else
                pl.lit(False, dtype=pl.Boolean)
                for j, x in enumerate(cols)
            ).collect().to_numpy()[0, :]
            if verbose:
                high_corr_cols = [cols[i] for i, is_low in enumerate(low_corr) if not is_low][:20]
                logger.info(f"Knock out round {i+1} is done. The feature selected is {feat}")
                logger.info(f"The following features are knocked out:\n{high_corr_cols}")

            surviving_indices &= low_corr
            pbar.update(1)
        if len(selected) >= output_size:
            break

    pbar.close()
    if len(selected) < k:
        logger.info(f"Found only {len(selected)}/{k} number of values because most of them "
                    "are highly correlated and the knock out rule eliminated most of them.")

    return selected

def mrmr_engine(
    df: PolarsFrame
    , k: int
    , relevance: dict[str, float]
    , strategy: MRMRSelectStrategy = "weighted_accum_corr"
    , **kwargs
) -> list[str]:
    '''
    A customizable MRMR Engine that runs the MRMR feature selection algorithm.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    k
        The number of features you want to select
    relevance
        A dictionary with keys being column names, values being relevance value of the
        feature.
    strategy
        How do you want the MRMR selection to be run? Valid strategies are `knock_out`,
        `accum_corr`, `weighted_accum_corr` or `custom`. Currently `custom` is not implemented
    
    kwargs
    ------
    If `strategy` is knock_out, you may optionally supply a corr_threshold keyword argument. The default
    value is 0.75 

    All builtin strategy has an optional verbose keyword argument for more details in the selection
    process.
    '''
    
    if len(relevance) == 0:
        return []
    
    cols = list(relevance.keys()) # Python dict is ordered.
    scores = np.fromiter((v for v in relevance.values()), dtype=np.float64)
    if (scores < 0).any():
        raise ValueError("Feature relevance scores must be all positive.")
    
    df_local = df.select(cols)
    if strategy == "accum_corr":
        return _accum_corr_mrmr(df_local, k, cols, scores, False, **kwargs)
    elif strategy == "weighted_accum_corr":
        return _accum_corr_mrmr(df_local, k, cols, scores, True, **kwargs)
    elif strategy == "knock_out":
        if "corr_threshold" not in kwargs:
            logger.warn("Found `corr_threshold` is None. Set to 0.75 by default.")
            kwargs["corr_threshold"] = 0.75
        
        return _knock_out_mrmr(df_local, k, cols, scores, **kwargs)
    elif strategy == "custom":
        return NotImplemented
    else:
        raise TypeError(f"The strategy {strategy} is not a valid MRMRSelectStrategy.")
    
def mrmr(
    df: PolarsFrame
    , target: str
    , k: int
    , cols: Optional[list[str]] = None
    , relevance: MRMRRelevance = "f"
    , mrmr_strategy: MRMRSelectStrategy = "weighted_accum_corr"
    , **kwargs
) -> list[str]:
    '''
    Run MRMR with builtin options.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe. Note for most options, df will be collected internally.
    target
        The target column
    k
        The number of features to output
    cols
        The columns to select from. If not given, all numerical ones will be used.
    relevance
        Oen of `f`, `lgbm`, `mis`, or `ks`. Right now only binary target is supported.
    mrmr_strategy
        How to run MRMR. One of `knock_out`, `accum_corr`, `weighted_accum_corr`    
    '''
    
    if cols is None:
        nums = get_numeric_cols(df, exclude=[target])
    else:
        _ = type_checker(df, cols, "numeric", "mrmr")
        nums = [c for c in cols if c != target]

    rel_dict = _mrmr_relevance(df, target, nums, relevance)
    return mrmr_engine(df, k, rel_dict, mrmr_strategy, **kwargs)

# -------------------------------------- End of MRMR -------------------------------------------


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

    if model_str in ("lgbm", "lightgbm"):
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**params)
    else:
        raise ValueError(f"The model {model_str} is not available.")
    # if model_str in ("logistic", "lr"):
    #     from sklearn.linear_model import LogisticRegression
    #     model = LogisticRegression(**params)
    # elif model_str in ("rf", "random_forest"):
    #     from sklearn.ensemble import RandomForestClassifier
    #     model = RandomForestClassifier(**params)
    # elif model_str in ("xgb", "xgboost"):
    #     from xgboost import XGBClassifier
    #     model = XGBClassifier(**params)
    # elif model_str in ("lgbm", "lightgbm"):
    #     from lightgbm import LGBMClassifier
    #     model = LGBMClassifier(**params)
    
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
        Only 'lgbm'
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
        me.log_loss(y_test, y_pred, check_binary=False),
        me.roc_auc(y_test, y_pred, check_binary=False)
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
        Only 'lgbm'
    params
        Parameters for the model
    n_comb
        We will run this for all n choose n_comb combinations of features
    '''
    features = get_numeric_cols(df, exclude=[target])
    fi = {f:[] for f in features}
    records = []
    pbar = tqdm(total=math.comb(len(features), n_comb), desc="Combinations", disable=dsds.NO_PROGRESS_BAR)
    df_keep = df.select(features + [target])
    for comb in combinations(features, r = n_comb):
        train, test = train_test_split(df_keep, train_frac)
        fc_rec, fi_rec = _fc_fi(model_str, params, target, comb, train, test) 
        records.append(fc_rec)
        for f, imp in zip(fc_rec[0], fi_rec):
            fi[f].append(imp)
        pbar.update(1)

    fc_summary = pl.from_records(records, schema=["combination", "logloss", "roc_auc"])
    fi_summary = pl.from_dict(fi)

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
        test_score += me.roc_auc(y, model.predict_proba(shuffled_df)[:, -1])

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
    encoded or transformed before calling this. Only works for binary classification and score = roc_auc for now.

    Parameters
    ----------
    df
        An eager Polars DataFrame
    target
        The target column
    model_str
        Only 'lgbm'
    params
        Parameters for the model
    k
        Permute the same feature k times
    '''
    features = df.columns
    features.remove(target)
    _ = type_checker(df, features, "numeric", "permutation_importance")
    estimator = _binary_model_init(model_str, params)
    X, y = df[features], df[target].to_numpy()
    estimator.fit(X, y)
    score = me.roc_auc(y, estimator.predict_proba(X)[:, -1])
    pbar = tqdm(total=len(features), desc="Permuting Features", disable=dsds.NO_PROGRESS_BAR)
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