from .prescreen import (
    discrete_inferral
    , get_numeric_cols
    , get_unique_count
)

from .type_alias import (
    PolarsFrame
    , MRMRStrategy
    , CPU_COUNT
    , clean_strategy_str
)
from .blueprint import( # Need this for Polars extension to work
    Blueprint
)

import polars as pl
import numpy as np
from typing import Any, Optional, Tuple
from scipy.spatial import KDTree
from scipy.special import fdtrc, psi
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def _conditional_entropy(
    df:pl.DataFrame
    , target:str
    , predictive:str
) -> Tuple[str, float]:

    cond_entropy = df.groupby((target, predictive)).agg(
        pl.count()
    ).with_columns(
        (pl.col("count").sum().over(predictive) / len(df)).alias("prob(predictive)"),
        (pl.col("count") / pl.col("count").sum()).alias("prob(target,predictive)")
    ).select(
        (-((pl.col("prob(target,predictive)")/pl.col("prob(predictive)")).log() 
           * pl.col("prob(target,predictive)")).sum()) # This is the conditional entropy.
    ).row(0)[0]

    return (predictive, cond_entropy)

# !!!NEED REVIEW FOR CORRECTNESS
def discrete_ig(
    df:pl.DataFrame
    , target:str
    , discrete_cols:Optional[list[str]] = None
    , n_threads:int = CPU_COUNT
) -> pl.DataFrame:
    '''The entropy here is "discrete entropy".

        Computes the information gain: Entropy(target) - Conditional_Entropy(target|c), where c is a column in 
        discrete_cols. For more information, please take a look at https://en.wikipedia.org/wiki/Entropy_(information_theory)

        Information gain defined in this way suffers from high cardinality (high uniqueness), and therefore a weighted 
        information gain is provided, weighted by (1 - unique_pct), where unique_pct represents the percentage of unique
         values in feature.

        Currently this only works for discrete columns. For continuous features vs discrete target, use mutual_info.

        Arguments:
            df: Eager frame only.
            target:
            discrete_cols: list of discrete columns.
            top_k: must be >= 0. If <= 0, the entire DataFrame will be returned.
            n_threads: 4, 8 ,16 will not make any real difference. But there is a difference between 0 and 4 threads. 
            
        Returns:
            a poalrs dataframe with information gain computed for each categorical column. 
    '''
    output = []
    discretes = []
    if isinstance(discrete_cols, list):
        discretes.extend(discrete_cols)
    else: # If discrete_cols is not passed, infer it
        discretes.extend(discrete_inferral(df, exclude=[target]))

    # Compute target entropy. This only needs to be done once.
    target_entropy = df.groupby(target).agg(
                        (pl.count()).alias("prob(target)") / len(df)
                    ).get_column("prob(target)").entropy()

    # Get unique count for selected columns. This is because higher unique percentage may skew information gain
    unique_count = get_unique_count(df.select(discretes)).with_columns(
        (pl.col("n_unique") / len(df)).alias("unique_pct")
    ).rename({"column":"feature"})

    with ThreadPoolExecutor(max_workers=n_threads) as ex: # 10% speed gain ? Need to retest this..
        pbar = tqdm(total=len(discretes), desc = "discrete_ig")
        for future in as_completed(ex.submit(_conditional_entropy, df, target, pred) for pred in discretes):
            ig = future.result()
            output.append(ig)
            pbar.update(1)

        pbar.close()

    return pl.from_records(output, schema=["feature", "conditional_entropy"])\
        .with_columns(
            pl.lit(target_entropy).alias("target_entropy"),
            (pl.lit(target_entropy) - pl.col("conditional_entropy")).alias("information_gain")
        ).join(unique_count, on="feature")\
        .select("feature", "target_entropy", "conditional_entropy", "unique_pct", "information_gain")\
        .with_columns(
            ((1 - pl.col("unique_pct")) * pl.col("information_gain")).alias("weighted_information_gain")
        )

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
    , random_state:int=42
    , n_threads:int=CPU_COUNT
) -> pl.DataFrame:
    '''Approximates mutual information (information gain) between the continuous variables and the target.

        This is essentially a "copy-and-paste" of the mutual_info_classif call in sklearn library. 
        There are a few important distinctions:

        1. This uses Scipy library's kdtree, instead of sklearn's kdtree and nearneighbors. 
        2. The use of Scipy enables us to use more cores. 
        3. There are less "checks" and "safeguards", meaning input data quality is expected to be "good".
        4. Conti_cols are supposed to be "continuous" variables. In sklearn's mutual_info_classif, if you input a dense 
            matrix X, it will always be treated as continuous, and if X is sparse, it will be treated as discrete.
        5. This method is based on method described the sources.

        Arguments:
            df:
            conti_cols: 
            target: list of discrete columns.
            n_neighbors:
            random_state: a random seed to generate small noise.
            n_threads: 
            
        Returns:

        Sources:
            (1). B. C. Ross “Mutual Information between Discrete and Continuous Data Sets”. PLoS ONE 9(2), 2014.\n
            (2). A. Kraskov, H. Stogbauer and P. Grassberger, “Estimating mutual information”. Phys. Rev. E 69, 2004.
            
    '''
    n = len(df)
    rng = np.random.default_rng(random_state)
    target_col = df.get_column(target).to_numpy().ravel()
    unique_targets = np.unique(target_col)
    all_masks = {}
    for t in unique_targets:
        all_masks[t] = target_col == t
        if np.sum(all_masks[t]) <= n_neighbors:
            raise ValueError(f"The target class {t} must have more than {n_neighbors} values in the dataset.")        

    estimates = []
    psi_n_and_k = psi(n) + psi(n_neighbors)
    for col in tqdm(conti_cols, desc = "Mutual Info"):
        c = df.get_column(col).cast(pl.Float64).to_numpy().reshape(-1,1)
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

    return pl.from_records([conti_cols, estimates], schema=["feature", "estimated_mi"])

# Selectors should always return target
def mutual_info_selector(
    df:PolarsFrame
    , target:str
    , n_neighbors:int=3
    , random_state:int=42
    , n_threads:int=CPU_COUNT
    , top_k:int = 50
) -> PolarsFrame:
    
    is_lazy = isinstance(df, pl.LazyFrame)
    if is_lazy:
        input_data:pl.DataFrame = df.collect()
    else:
        input_data:pl.DataFrame = df

    nums = get_numeric_cols(input_data, exclude=[target])
    complement = [f for f in input_data.columns if f not in nums]

    mi_scores = mutual_info(input_data, target, nums, n_neighbors, random_state, n_threads)\
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
    , num_cols:Optional[list[str]]=None
) -> pl.DataFrame:
    '''Computes ANOVA one way test, the f value/score and the p value. 
        Equivalent to f_classif in sklearn.feature_selection, but is more dataframe-friendly, 
        and performs better on bigger data.

        Arguments:
            df: input Polars dataframe.
            target: the target column.
            num_cols: if provided, will run the ANOVA one way test for each column in num_cols. If none,
                will try to infer from df according to data types. Note that num_cols should be numeric!

        Returns:
            a polars dataframe with f score and p value.
    
    '''
    num_list = []
    if isinstance(num_cols, list):
        num_list.extend(num_cols)
    else:
        num_list.extend(get_numeric_cols(df, exclude=[target]))

    # Get average within group and sample variance within group.
    ## Could potentially replace this with generators instead of lists. Not sure how impactful that would be... Probably no diff.
    step_one_expr:list[pl.Expr] = [pl.count().alias("cnt")] # get cnt, and avg within classes
    step_two_expr:list[pl.Expr] = [] # Get average for each column
    step_three_expr:list[pl.Expr] = [] # Get "f score" (without some normalizer, see below)
    # Minimize the amount of looping and str concating in Python. Use Exprs as much as possible.
    for n in num_list:
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
    return pl.from_records((num_list, f_values, p_values), schema=["feature","f_value","p_value"])

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

#---- Below is MRMR

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
    , num_cols:Optional[list[str]] = None
    , strategy: MRMRStrategy = "fscore"
    , params:Optional[dict[str,Any]] = None
    , low_memory:bool=False
) -> list[str]:
    '''Implements MRMR. Will add a few more strategies in the future. Likely only strategies for numerators
        , aka relevance. Right now xgb, lgbm and rf strategies only work for classification problems.

        Currently this only supports classification.

        Arguments:
            df:
            target:
            k:
            num_cols:
            strategy: by default, f-score will be used.
            params: if a RF/XGB strategy is selected, params is a dict of parameters for the model.
            low_memory: 

        Returns:
            pl.DataFrame of features and the corresponding ranks according to the mrmr_algo
    
    '''

    num_list = []
    if isinstance(num_cols, list):
        num_list.extend(num_cols)
    else:
        num_list.extend(get_numeric_cols(df, exclude=[target]))

    s = clean_strategy_str(strategy)
    scores = _mrmr_underlying_score(df
                                    , target = target
                                    , num_list = num_list
                                    , strategy = s
                                    , params = {} if params is None else params
                                    )

    if low_memory:
        df_local = df.select(num_list)
    else: # this could potentially double memory usage. so I provided a low_memory flag.
        df_local = df.select(num_list).with_columns(
            (pl.col(f) - pl.col(f).mean())/pl.col(f).std() for f in num_list
        ) # Note that if we get a const column, the entire column will be NaN

    output_size = min(k, len(num_list))
    print(f"Found {len(num_list)} total features to select from. Proceeding to select top {output_size} features.")
    cumulating_abs_corr = np.zeros(len(num_list)) # For each feature at index i, we keep a cumulating sum
    top_idx = np.argmax(scores)
    selected = [num_list[top_idx]]
    pbar = tqdm(total=output_size, desc = f"MRMR, {s}")
    pbar.update(1)
    for j in range(1, output_size): 
        argmax = -1
        current_max = -1
        last_selected_col = df_local.drop_in_place(selected[-1])
        if low_memory: # normalize if in low memory mode.
            last_selected_col = (last_selected_col - last_selected_col.mean())/last_selected_col.std()
        for i,f in enumerate(num_list):
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

        selected.append(num_list[argmax])
        pbar.update(1)

    pbar.close()
    print("Output is sorted in order of selection (relevance).")
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

    num_cols = get_numeric_cols(input_data, exclude=[target])
    # Non-numerical columns cannot be analyzed by mrmr. So add back in the end.
    complement = [f for f in input_data.columns if f not in num_cols]

    s = clean_strategy_str(strategy)
    selected = mrmr(input_data, target, top_k, num_cols, s, params, low_memory)

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
        variables, here we use a simpler knock out rule based on absolute correlation.

        Inspired by the package Featurewiz and its creator.
    
    '''
    
    num_list = []
    if isinstance(num_cols, list):
        num_list.extend(num_cols)
    else:
        num_list.extend(get_numeric_cols(df, exclude=[target]))

    s = clean_strategy_str(strategy)
    scores = _mrmr_underlying_score(df
                                    , target = target
                                    , num_list = num_list
                                    , strategy = s
                                    , params = {} if params is None else params)

    # Set up
    low_corr = np.abs(df[num_list].corr().to_numpy()) < corr_threshold
    surviving_indices = np.full(shape=len(num_list), fill_value=True) # an array of booleans
    scores = list(enumerate(scores))
    scores.sort(key=lambda x:x[1], reverse=True)
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

    print("Output is sorted in order of selection (relevance).")
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
                    



    