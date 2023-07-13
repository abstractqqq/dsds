from .prescreen import (
    discrete_inferral
    , check_binary_target
    , get_numeric_cols
    , get_unique_count
    , get_string_cols
)

from .type_alias import (
    PolarsFrame
    , MRMRStrategy
    , BinaryModels
    , CPU_COUNT
    , clean_strategy_str
)
from .blueprint import( # Need this for Polars extension to work
    Blueprint
)
from .sample import (
    train_test_split
)
from .metrics import (
    logloss
    , roc_auc
)
import polars as pl
import numpy as np
from typing import Any, Optional, Tuple
from scipy.spatial import KDTree
from scipy.special import fdtrc, psi
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm
import math
from itertools import combinations

logger = logging.getLogger(__name__)

def discrete_ig(
    df:pl.DataFrame
    , target:str
    , cols:Optional[list[str]] = None
) -> pl.DataFrame:

    if isinstance(cols, list):
        discretes = cols
    else: # If discrete_cols is not passed, infer it
        discretes = discrete_inferral(df, exclude=[target])

    # Compute target entropy. This only needs to be done once.
    target_entropy = df.groupby(target).agg(
                        (pl.count()).alias("prob(target)") / len(df)
                    ).get_column("prob(target)").entropy()

    # Get unique count for selected columns. This is because higher unique percentage may skew information gain
    unique_count = get_unique_count(df.select(discretes)).with_columns(
        (pl.col("n_unique") / len(df)).alias("unique_pct")
    ).rename({"column":"feature"})

    conditional_entropy = (
        df.lazy().groupby(target, pred).agg(
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

    return pl.concat(conditional_entropy)\
        .with_columns(
            target_entropy = pl.lit(target_entropy),
            information_gain = pl.max(pl.lit(target_entropy) - pl.col("conditional_entropy"), 0)
        ).join(unique_count, on="feature")\
        .select("feature", "target_entropy", "conditional_entropy", "unique_pct", "information_gain")\
        .with_columns(
            weighted_information_gain = (1 - pl.col("unique_pct")) * pl.col("information_gain")
        ).collect()

discrete_mi = discrete_ig

def discrete_ig_selector(
    df:PolarsFrame
    , target:str
    , top_k:int
    , n_threads:int = CPU_COUNT
) -> PolarsFrame:
    
    discrete_cols = discrete_inferral(df, exclude=[target])
    is_lazy = isinstance(df, pl.LazyFrame)
    if is_lazy:
        input_data:pl.DataFrame = df.collect()
    else:
        input_data:pl.DataFrame = df

    ig = discrete_ig(input_data, target, discrete_cols, n_threads)\
            .top_k(by="information_gain", k = top_k)

    complement = [f for f in input_data.columns if f not in discrete_cols]
    selected = ig.get_column("feature").to_list()
    print(f"Selected {len(selected)} features. There are {len(complement)} columns the algorithm "
        "cannot process. They are also returned.")

    if is_lazy:
        return df.blueprint.select(selected + complement)
    return df.select(selected + complement)

def mutual_info(
    df:pl.DataFrame
    , target:str
    , conti_cols:list[str]
    , n_neighbors:int=3
    , seed:int=42
    , n_threads:int=CPU_COUNT
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
    n_threads
        The number of threads used in scipy's Kdtree

    Sources
    -------
        (1). B. C. Ross “Mutual Information between Discrete and Continuous Data Sets”. PLoS ONE 9(2), 2014.\n
        (2). A. Kraskov, H. Stogbauer and P. Grassberger, “Estimating mutual information”. Phys. Rev. E 69, 2004. 
    '''
    n = len(df)
    rng = np.random.default_rng(seed)
    target_col = df.get_column(target).to_numpy().ravel()
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
            dd, _ = kd1.query(c_masked, k = n_neighbors + 1, workers=n_threads)
            radius[mask] = np.nextafter(dd[:, -1], 0)
            label_counts[mask] = np.sum(mask)

        kd2 = KDTree(data=c, leafsize=40) 
        m_all = kd2.query_ball_point(c, r = radius, return_length=True, workers=n_threads)
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
    , seed:int=42
    , n_threads:int=CPU_COUNT
    , top_k:int = 50
) -> PolarsFrame:
    '''
    A selector based on the mutual_info feature selection method.

    This 
    
    '''
    
    is_lazy = isinstance(df, pl.LazyFrame)
    if is_lazy:
        input_data:pl.DataFrame = df.collect()
    else:
        input_data:pl.DataFrame = df

    nums = get_numeric_cols(input_data, exclude=[target])
    complement = [f for f in input_data.columns if f not in nums]

    mi_scores = mutual_info(input_data, target, nums, n_neighbors, seed, n_threads)\
                .top_k(by="estimated_mi", k = top_k)

    selected = mi_scores.get_column("feature").to_list()
    print(f"Selected {len(selected)} features. There are {len(complement)} columns the algorithm "
        "cannot process. They are also returned.")

    if is_lazy:
        return df.blueprint.select(selected + complement)
    return df.select(selected + complement)

def _f_score(
    df:pl.DataFrame
    , target:str
    , num_list:list[str]
) -> np.ndarray:
    '''Same as f_classif, but returns a numpy array of f scores only.'''
    
    # See comments in f_classif
    step_one_expr:list[pl.Expr] = [pl.count().alias("cnt")]
    step_two_expr:list[pl.Expr] = []
    step_three_expr:list[pl.Expr] = []
    for n in num_list:
        n_avg:str = n + "_avg"
        n_tavg:str = n + "_tavg"
        n_var:str = n + "_var"
        step_one_expr.append(
            pl.col(n).mean().alias(n_avg)
        )
        step_one_expr.append(
            pl.col(n).var(ddof=0).alias(n_var)
        )
        step_two_expr.append(
            (pl.col(n_avg).dot(pl.col("cnt")) / len(df)).alias(n_tavg)
        )
        step_three_expr.append(
            (pl.col(n_avg) - pl.col(n_tavg)).pow(2).dot(pl.col("cnt"))/ pl.col(n_var).dot(pl.col("cnt"))
        )

    ref = df.groupby(target).agg(step_one_expr)
    n_samples = len(df)
    n_classes = len(ref)
    df_btw_class = n_classes - 1 
    df_in_class = n_samples - n_classes
    # This is f-score, score in the order of num_list
    return ref.with_columns(step_two_expr).select(step_three_expr)\
            .to_numpy().ravel() * (df_in_class / df_btw_class)

def f_classif(
    df:pl.DataFrame
    , target:str
    , cols:Optional[list[str]]=None
) -> pl.DataFrame:
    '''Computes ANOVA one way test, the f value/score and the p value. 
        Equivalent to f_classif in sklearn.feature_selection, but is more dataframe-friendly, 
        and performs better on bigger data.

        Arguments:
            df: input Polars dataframe.
            target: the target column.
            cols: if provided, will run the ANOVA one way test for each column in num_cols. If none,
                will try to infer from df according to data types. Note that num_cols should be numeric!

        Returns:
            a polars dataframe with f score and p value.
    
    '''
    if isinstance(cols, list):
        nums = cols
    else:
        nums = get_numeric_cols(df, exclude=[target])

    # Get average within group and sample variance within group.
    ## Could potentially replace this with generators instead of lists. Not sure how impactful that would be... Probably no diff.
    step_one_expr:list[pl.Expr] = [pl.count().alias("cnt")] # get cnt, and avg within classes
    step_two_expr:list[pl.Expr] = [] # Get average for each column
    step_three_expr:list[pl.Expr] = [] # Get "f score" (without some normalizer, see below)
    # Minimize the amount of looping and str concating in Python. Use Exprs as much as possible.
    for n in nums:
        n_avg:str = n + "_avg" # avg within class
        n_tavg:str = n + "_tavg" # true avg / overall average
        n_var:str = n + "_var" # var within class
        step_one_expr.append(
            pl.col(n).mean().alias(n_avg)
        )
        step_one_expr.append(
            pl.col(n).var(ddof=0).alias(n_var) # ddof = 0 so that we don't need to compute pl.col("cnt") - 1
        )
        step_two_expr.append( # True average of this column, reduce the amount of repeated computation.
            # by using n_avg column dotted with cnt
            (pl.col(n_avg).dot(pl.col("cnt")) / len(df)).alias(n_tavg)
        )
        step_three_expr.append(
            # Between class var (without diving by df_btw_class) / Within class var (without dividng by df_in_class) 
            (pl.col(n_avg) - pl.col(n_tavg)).pow(2).dot(pl.col("cnt"))/ pl.col(n_var).dot(pl.col("cnt"))
        )

    # Get in class average and var
    ref = df.groupby(target).agg(step_one_expr)
    n_samples = len(df)
    n_classes = len(ref)
    df_btw_class = n_classes - 1 
    df_in_class = n_samples - n_classes
    
    f_values = ref.with_columns(step_two_expr).select(step_three_expr)\
            .to_numpy().ravel() * (df_in_class / df_btw_class)
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
    
    is_lazy = isinstance(df, pl.LazyFrame)
    if is_lazy:
        input_data:pl.DataFrame = df.collect()
    else:
        input_data:pl.DataFrame = df

    nums = get_numeric_cols(input_data, exclude=[target])
    # Non-numerical columns cannot be analyzed by mrmr. So add back in the end.
    complement = [f for f in input_data.columns if f not in nums]

    scores = _f_score(input_data, target, nums)
    temp_df = pl.DataFrame({"feature":nums, "fscore":scores}).top_k(
        by = "fscore", k = top_k
    )
    selected = temp_df.get_column("feature").to_list()
    
    print(f"Selected {len(selected)} features. There are {len(complement)} columns the algorithm "
    "cannot process. They are also returned.")

    if is_lazy:
        return df.blueprint.select(selected + complement)
    return df.select(selected + complement)

def _mrmr_underlying_score(
    df:pl.DataFrame
    , target:str
    , num_list:list[str]
    , strategy:MRMRStrategy
    , params:dict[str,Any]
) -> np.ndarray:
    
    print(f"Running {strategy} to determine feature relevance...")
    s = clean_strategy_str(strategy)
    if s in ("fscore", "f", "f_score"):
        scores = _f_score(df, target, num_list)
    elif s in ("rf", "random_forest"):
        from sklearn.ensemble import RandomForestClassifier
        print("Random forest is not deterministic by default. Results may vary.")
        rf = RandomForestClassifier(**params)
        rf.fit(df[num_list].to_numpy(), df[target].to_numpy().ravel())
        scores = rf.feature_importances_
    elif s in ("xgb", "xgboost"):
        from xgboost import XGBClassifier
        print("XGB is not deterministic by default. Results may vary.")
        xgb = XGBClassifier(**params)
        xgb.fit(df[num_list].to_numpy(), df[target].to_numpy().ravel())
        scores = xgb.feature_importances_
    elif s in ("mis", "mutual_info_score"):
        scores = mutual_info(df, conti_cols=num_list, target=target).get_column("estimated_mi").to_numpy().ravel()
    elif s in ("lgbm", "lightgbm"):
        from lightgbm import LGBMClassifier
        print("LightGBM is not deterministic by default. Results may vary.")
        lgbm = LGBMClassifier(**params)
        lgbm.fit(df[num_list].to_numpy(), df[target].to_numpy().ravel())
        scores = lgbm.feature_importances_
    else: # Pythonic nonsense
        raise ValueError(f"The strategy {strategy} is not a valid MRMR Strategy.")
    
    return scores

def mrmr(
    df:pl.DataFrame
    , target:str
    , k:int
    , cols:Optional[list[str]] = None
    , strategy: MRMRStrategy = "fscore"
    , params:Optional[dict[str,Any]] = None
    , low_memory:bool=False
) -> list[str]:
    '''
    Implements MRMR. Will add a few more strategies in the future. Likely only strategies for numerators
    , aka relevance. Right now xgb, lgbm and rf strategies only work for classification problems.

    Currently this only supports classification.

    Parameters
    ----------
    df
        An eager Polars Dataframe
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
    low_memory
        Whether to do some computation all at once, which uses more memory at once, or do some 
        computation when needed, which uses less memory at any given time.

    Returns
    -------
        A list of top k features
    
    '''
    if isinstance(cols, list):
        nums = cols
    else:
        nums = get_numeric_cols(df, exclude=[target])

    s = clean_strategy_str(strategy)
    scores = _mrmr_underlying_score(df
        , target = target
        , num_list = nums
        , strategy = s
        , params = {} if params is None else params
    )

    if low_memory:
        df_local = df.select(nums)
    else: # this could potentially double memory usage. so I provided a low_memory flag.
        df_local = df.select(nums).with_columns(
            (pl.col(f) - pl.col(f).mean())/pl.col(f).std() for f in nums
        ) # Note that if we get a const column, the entire column will be NaN

    output_size = min(k, len(nums))
    print(f"Found {len(nums)} total features to select from. Proceeding to select top {output_size} features.")
    cumulating_abs_corr = np.zeros(len(nums)) # For each feature at index i, we keep a cumulating sum
    top_idx = np.argmax(scores)
    selected = [nums[top_idx]]
    pbar = tqdm(total=output_size, desc = f"MRMR, {s}")
    pbar.update(1)
    for j in range(1, output_size): 
        argmax = -1
        current_max = -1
        last_selected_col = df_local.drop_in_place(selected[-1])
        if low_memory: # normalize if in low memory mode.
            last_selected_col = (last_selected_col - last_selected_col.mean())/last_selected_col.std()
        for i,f in enumerate(nums):
            if f not in selected:
                # Left = cumulating sum of abs corr
                # Right = abs correlation btw last_selected and f
                candidate_col = df_local.get_column(f)
                if low_memory: # normalize if in low memory mode.
                    candidate_col = (candidate_col - candidate_col.mean())/candidate_col.std()

                a = (last_selected_col*candidate_col).mean()
                # In the rare case this calculation yields a NaN, we punish by adding 1.
                # Otherwise, proceed as usual. +1 is a punishment because
                # |corr| can be at most 1. So we are enlarging the denominator, thus reducing the score.
                cumulating_abs_corr[i] += 1 if np.isnan(a) else np.abs(a)
                denominator = cumulating_abs_corr[i]/j 
                new_score = scores[i] / denominator
                if new_score > current_max:
                    current_max = new_score
                    argmax = i
        selected.append(nums[argmax])
        pbar.update(1)

    pbar.close()
    print("Output is sorted in order of selection (max relevance min redundancy).")
    return selected

def mrmr_selector(
    df:PolarsFrame
    , target:str
    , top_k:int
    , strategy:MRMRStrategy = "fscore"
    , params:Optional[dict[str,Any]] = None
    , low_memory:bool=False
) -> PolarsFrame:

    is_lazy = isinstance(df, pl.LazyFrame)
    if is_lazy:
        input_data:pl.DataFrame = df.collect()
    else:
        input_data:pl.DataFrame = df

    nums = get_numeric_cols(input_data, exclude=[target])
    # Non-numerical columns cannot be analyzed by mrmr. So add back in the end.
    complement = [f for f in input_data.columns if f not in nums]
    s = clean_strategy_str(strategy)
    selected = mrmr(input_data, target, top_k, nums, s, params, low_memory)

    print(f"Selected {len(selected)} features. There are {len(complement)} columns the algorithm "
          "cannot process. They are also returned.")
    
    if is_lazy:
        return df.blueprint.select(selected + complement)
    return df.select(selected + complement)

def knock_out_mrmr(
    df:pl.DataFrame
    , target:str
    , k:int 
    , num_cols:Optional[list[str]] = None
    , strategy:MRMRStrategy = "fscore"
    , corr_threshold:float = 0.7
    , params:Optional[dict[str,Any]] = None
) -> list[str]:
    '''
    Essentially the same as vanilla MRMR. Instead of using sum(abs(corr)) to "weigh down" correlated 
    variables, here we use a simpler knock out rule based on absolute correlation. We go down the list
    according to importance, take top one, knock out all other features that are highly correlated with
    it, take the next top feature that has not been knocked out, continue, until we pick enough features
    or there is no feature left.

    Inspired by the package Featurewiz and its creator.


    
    '''
    if isinstance(num_cols, list):
        num_list = num_cols
    else:
        num_list = get_numeric_cols(df, exclude=[target])

    s = clean_strategy_str(strategy)
    scores = _mrmr_underlying_score(df
        , target = target
        , num_list = num_list
        , strategy = s
        , params = {} if params is None else params
    )

    # Set up
    low_corr = np.abs(df[num_list].corr().to_numpy()) < corr_threshold
    surviving_indices = np.full(shape=len(num_list), fill_value=True) # an array of booleans
    scores = sorted(enumerate(scores), key=lambda x:x[1], reverse=True)
    selected = []
    count = 0
    output_size = min(k, len(num_list))
    pbar = tqdm(total=output_size)
    # Run the knock outs
    for i, _ in scores:
        if surviving_indices[i]:
            selected.append(num_list[i])
            surviving_indices = surviving_indices & low_corr[:,i]
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
    , strategy:MRMRStrategy = "fscore"
    , corr_threshold:float = 0.7
    , params:Optional[dict[str,Any]] = None
) -> PolarsFrame:

    is_lazy = isinstance(df, pl.LazyFrame)
    if isinstance(df, pl.LazyFrame):
        input_data:pl.DataFrame = df.collect()
    else:
        input_data:pl.DataFrame = df

    num_cols = get_numeric_cols(df, exclude=[target])
    # Non-numerical columns cannot be analyzed by mrmr. So add back in the end.
    complement = [f for f in df.columns if f not in num_cols]

    s = clean_strategy_str(strategy)
    selected = knock_out_mrmr(input_data, target, top_k, num_cols, s, corr_threshold, params)
    print(f"Selected {len(selected)} features. There are {len(complement)} columns the algorithm "
        "cannot process. They are also returned.")
    
    if is_lazy:
        return df.blueprint.select(selected + complement)
    return df.select(selected + complement)

# Selectors for the methods below are not yet implemented

# Create a numeric + string version of woe_iv in the future
def woe_iv_cat(
    df:PolarsFrame
    , target:str
    , cols:Optional[list[str]]=None
    , min_count:float = 1.
    , check_binary:bool = True
) -> pl.DataFrame:
    '''
        Arguments:
            df: ..
            target: ..
            cols: a list of string or highly discrete numeric columns. Currently woe_iv for continuous values
            is not implemenented. If not provided, it will use only string columns.
            min_count: a regularization term that prevents ln(0). This is the same as 
            category_encoders package's regularization parameter.
            check_binary: whether to check if target is binary or not
    '''
    if isinstance(cols, list):
        input_cols = cols
    else:
        input_cols = get_string_cols(df)

    if check_binary:
        if not check_binary_target(df, target):
            raise ValueError("Target is not binary or not properly encoded.")

    results = (
        df.lazy().groupby(s).agg(
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
    return pl.concat(results).collect()

def _binary_model_init(
    model:BinaryModels
    , params: dict[str, Any]
):
    if "n_jobs" not in params:
        params["n_jobs"] = -1

    if model in ("logistic", "lr"):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**params)
    elif model in ("rf", "random_forest"):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
    elif model in ("xgb", "xgboost"):
        from xgboost import XGBClassifier
        model = XGBClassifier(**params)
    elif model in ("lgbm", "lightgbm"):
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**params)
    else:
        raise ValueError(f"The model {model} is not available.")
    
    return model

def _ebfs(
    model:str
    , params:dict[str, Any]
    , target:str
    , comb: Tuple
    , train: pl.DataFrame
    , test:pl.DataFrame
)-> Tuple[Tuple[Tuple, float, float, float], np.ndarray]:
    estimator = _binary_model_init(model, params)
    _ = estimator.fit(train.select(comb), train[target])
    y_pred = estimator.predict_proba(test.select(comb))[:,1]
    y_test = test[target].to_numpy()
    rec = (
        comb,
        logloss(y_test, y_pred, check_binary=False),
        roc_auc(y_test, y_pred, check_binary=False)
    )
    if model in ("lr", "logistic"):
        importances = np.abs(estimator.coef_).ravel()
    else:
        importances = estimator.feature_importances_

    return rec, importances

# ebfs: stands for Exhaustive Binary Feature Selection
def ebfs(
    df:pl.DataFrame
    , target:str
    , model:BinaryModels
    , params:dict[str, Any]
    , n_comb: int = 3
    , train_frac:float = 0.75
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''Suppose we have n features and n_comb = 2. This method will select all (n choose 2) 
    combinations of features, split dataset into a train and a test, train a model on train, 
    and compute feature importance and roc_auc and logloss, and then finally put 
    everything into two separate dataframes, the first of which will contain the feature
    combinations and model performances, and the second will contain the min, avg, max and var of 
    feature importance of each feature in all its occurences in the training rounds.

    This method will be extremely slow if (n choose n_comb) is a big number. All numerical columns 
    will be taken as potential features. Please encode the string columns if you want to use them
    as features here.

    Future Improvement? Record progress and skip through results.

    If n_jobs is not provided in params, it will be defaulted to -1.

        Arguments:
            df: an eager Polars DataFrame
            target: target column
            model: one of 'logistic', 'lr', 'lightgbm', 'lgbm', 'xgboost', 'xgb', 'random_forest', 'rf'
            params: parameters for the model.
            n_comb: n choose n_comb

        Returns:
            a feature combination summary, a feature importance summary
    '''
    
    features = get_numeric_cols(df, exclude=[target])
    train, test = train_test_split(df.select(features + [target]), train_frac)
    fi = {f:[] for f in features}
    records = []
    pbar = tqdm(total=math.comb(len(features), n_comb), desc="Total Combinations")
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as ex:
        futures = (
            ex.submit(
                _ebfs
                , model
                , params
                , target
                , comb
                , train
                , test
            )
            for comb in combinations(features, r=n_comb)
        )
        for f in as_completed(futures):
            fc_rec, fi_rec = f.result()
            records.append(fc_rec)
            for f, i in zip(fc_rec[0], fi_rec):
                fi[f].append(i)
            pbar.update(1)

    fc_summary = pl.from_records(records, schema=["combination", "logloss", "roc_auc"])
    stats = [
        (f, len(fi[f]), np.min(fi[f]), np.mean(fi[f]), np.max(fi[f]), np.std(fi[f])) for f in fi
    ]
    fi_summary = pl.from_records(stats, schema=["feature", "occurrences", "min", "mean", "max", "std"])
    pbar.close()
    return fc_summary, fi_summary

def ebfs_fc_filter(
    fc: pl.DataFrame
    , logloss_threshold:float
    , roc_auc_threshold:float
) -> list[str]:
    '''A filter method based on the feature combination result of ebfs. First, 

        Arguments:
            fc: the feature combination result from ebfs
            logloss_threshold: the maximum logloss for the combination to be kept
            roc_auc_threshold: the minumum roc_auc for the combination to be kept

        Returns:
            a list of string representing all features in the combinations after filtering
    '''
    return fc.filter(
        pl.col("logloss") <= logloss_threshold
        & pl.col("roc_auc") >= roc_auc_threshold
    ).get_column("combination").explode().unique().to_list()


    