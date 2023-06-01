import os
import polars as pl
import numpy as np
from enum import Enum
from typing import Final, Any
from scipy.special import fdtrc
from concurrent.futures import ThreadPoolExecutor, as_completed
from .eda_prescreen import get_string_cols, get_numeric_cols, get_unique_count
from tqdm import tqdm

CPU_COUNT:Final[int] = os.cpu_count()

def _conditional_entropy(df:pl.DataFrame, target:str, predictive:str) -> pl.DataFrame:
    temp = df.groupby(predictive).agg(
        pl.count().alias("prob(predictive)")
    ).with_columns(
        pl.col("prob(predictive)") / len(df)
    )

    return df.groupby((target, predictive)).agg(
        pl.count()
    ).with_columns(
        (pl.col("count") / pl.col("count").sum()).alias("prob(target,predictive)")
    ).join(
        temp, on=predictive
    ).select((
        pl.lit(predictive).alias("feature"),
        (-((pl.col("prob(target,predictive)")/pl.col("prob(predictive)")).log() * pl.col("prob(target,predictive)")).sum()).alias("conditional_entropy")
    ))

# NEED REVIEW OF CORRECTNESS
def information_gain(df:pl.DataFrame, target:str
    , discrete_cols:list[str] = None
    , top_k:int = 0
    , n_threads:int = CPU_COUNT) -> pl.DataFrame:
    '''
        Computes the information gain: Entropy(target) - Conditional_Entropy(target|c), where c is a column in discrete_cols.
        For more information, please take a look at https://en.wikipedia.org/wiki/Entropy_(information_theory)

        Information gain defined in this way suffers from high cardinality (high uniqueness), and therefore a weighted information
        gain is provided, weighted by (1 - unique_pct), where unique_pct represents the percentage of unique values in feature.

        Currently this only works for discrete columns and no method for continuous column is implemented yet.

        Arguments:
            df:
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
    else: # If cat_cols is not passed, infer it
        discretes.extend(get_string_cols(df, exclude=[target]))

    if len(discretes) == 0:
        print(f"No columns are provided or can be inferred.")
        print("Returned empty dataframe.")
        return pl.DataFrame()
    
    # Compute target entropy. This only needs to be done once.
    target_entropy = df.groupby(target).agg(
                        (pl.count()).alias("prob(target)") / len(df)
                    ).get_column("prob(target)").entropy() 

    # Get unique count for selected columns. This is because higher unique percentage may skew information gain
    unique_count = get_unique_count(df.select(discretes)).with_columns(
        (pl.col("n_unique") / len(df)).alias("unique_pct")
    ).rename({"column":"feature"})

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = (ex.submit(_conditional_entropy, df, target, predictive) for predictive in discretes)
        with tqdm(total=len(discretes)) as pbar:
            for res in as_completed(futures):
                ig = res.result()
                output.append(ig)
                pbar.update(1)

    output = pl.concat(output).with_columns((
        pl.lit(target_entropy).alias("target_entropy"),
        (pl.lit(target_entropy) - pl.col("conditional_entropy")).alias("information_gain")
    )).join(unique_count, on="feature")\
        .select(("feature", "target_entropy", "conditional_entropy", "unique_pct", "information_gain"))\
        .with_columns(
            ((1 - pl.col("unique_pct")) * pl.col("information_gain")).alias("weighted_information_gain")
        ).sort("information_gain", descending=True)

    if top_k <= 0:
        return output 
    else:
        return output.limit(top_k)


def _f_score(df:pl.DataFrame, target:str, num_list:list[str]) -> np.ndarray:
    '''
        Same as f_classification, but returns a numpy array of f scores only. 
        This is used in algorithms like MRMR for easier-to-work-with output datatype.

    '''
    
    # Get average within group and sample variance within group.
    ## Could potentially replace this with generators instead of lists. Not sure how impactful that would be... Probably no diff.
    step_one_expr:list[pl.Expr] = [pl.count().alias("cnt")] # get cnt, and avg within classes
    step_two_expr:list[pl.Expr] = [] # Get average for each column
    step_three_expr:list[pl.Expr] = [] # Get "f score" (without some normalizer, see below)
    # Minimize the amount of looping and str concating in Python. Use Exprs as much as possible.
    for n in num_list:
        n_avg:str = n + "_avg" # avg within class
        n_tavg:str = n + "_tavg" # true avg / absolute average
        n_var:str = n + "_var" # var within class
        step_one_expr.append(
            pl.col(n).mean().alias(n_avg)
        )
        step_one_expr.append(
            pl.col(n).var(ddof=0).alias(n_var) # ddof = 0 so that we don't need to compute pl.col("cnt") - 1
        )
        step_two_expr.append( # True average of this column
            (pl.col(n_avg).dot(pl.col("cnt")) / len(df)).alias(n_tavg)
        )
        step_three_expr.append(
            # Between class var (without diving by df_btw_class) / Within class var (without dividng by df_in_class) 
            (pl.col(n_avg) - pl.col(n_tavg)).pow(2).dot(pl.col("cnt"))/ pl.col(n_var).dot(pl.col("cnt"))
        )

    # Get in class average and var
    ref = df.groupby(target).agg(step_one_expr)
    n_samples = np.float64(len(df))
    n_classes = np.float64(len(ref))
    df_btw_class = n_classes - 1 
    df_in_class = n_samples - n_classes
    
    f_score = ref.with_columns(step_two_expr).select(step_three_expr)\
            .to_numpy().ravel() * (df_in_class / df_btw_class)
    
    return f_score

def f_classification(df:pl.DataFrame, target:str, num_cols:list[str]=None) -> pl.DataFrame:
    '''
        Computes ANOVA one way test, the f value/score and the p value. 
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
        n_tavg:str = n + "_tavg" # true avg / absolute average
        n_var:str = n + "_var" # var within class
        step_one_expr.append(
            pl.col(n).mean().alias(n_avg)
        )
        step_one_expr.append(
            pl.col(n).var(ddof=0).alias(n_var) # ddof = 0 so that we don't need to compute pl.col("cnt") - 1
        )
        step_two_expr.append( # True average of this column
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
    # At this point, f_values should be a pretty small dataframe. Cast to numpy, so that fdtrc can process it properly.

    p_values = fdtrc(df_btw_class, df_in_class, f_values) # get p values 
    return pl.from_records([num_list, f_values, p_values], schema=["feature","f_value","p_value"])


class MRMR_STRATEGY(Enum):
    F_SCORE = 1
    RF = 2
    XGB = 3

def mrmr(df:pl.DataFrame, target:str, k:int, num_cols:list[str]=None
        , strategy:MRMR_STRATEGY=MRMR_STRATEGY.F_SCORE
        , params:dict[str:Any]={}) -> pl.DataFrame:
    '''
        Implements FCQ MRMR. Will add a few more strategies in the future. (Likely only strategies for numerators)
        See https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b
        for more information.

        Currently this only supports classification.

        Arguments:
            df:
            target:
            k:
            num_cols:
            strategy: by default, f-score will be used.
            params: if a RF/XGB strategy is selected, params is a dict of parameters for the model.

        Returns:
            pl.DataFrame of features and the corresponding ranks according to the mrmr_algo
    
    '''


    # from sklearn.ensemble import RandomForestClassifier

    num_list = []
    if isinstance(num_cols, list):
        num_list.extend(num_cols)
    else:
        num_list.extend(get_numeric_cols(df, exclude=[target]))

    if strategy == MRMR_STRATEGY.F_SCORE:
        scores = _f_score(df, target, num_list)
    elif strategy == MRMR_STRATEGY.RF:
        from sklearn.ensemble import RandomForestClassifier
        print("Random forest is not deterministic by default. Results may vary.")
        rf = RandomForestClassifier(**params)
        rf.fit(df[num_list].to_numpy(), df[target].to_numpy().ravel())
        scores = rf.feature_importances_
    elif strategy == MRMR_STRATEGY.XGB:
        from xgboost import XGBClassifier
        print("XGB is not deterministic by default. Results may vary.")
        xgb = XGBClassifier(**params)
        xgb.fit(df[num_list].to_numpy(), df[target].to_numpy().ravel())
        scores = xgb.feature_importances_
    else: # Pythonic nonsense
        scores = _f_score(df, target, num_list)

    # FYI. This part can be removed.
    importance = pl.from_records(list(zip(num_list, scores)), schema=["feature", str(strategy)])\
                    .top_k(by=str(strategy), k=5)
    print(f"Top 5 feature importance is (by {strategy}):\n{importance}")

    df_scaled = df.select(num_list).with_columns(
        (pl.col(f) - pl.col(f).mean())/pl.col(f).std() for f in num_list
    ) # Note that if we get a const column, the entire column will be NaN

    output_size = min(k, len(num_list))
    print(f"Found {len(num_list)} total features to select from. Proceeding to select top {output_size} features.")
    cumulating_abs_corr = np.zeros(len(num_list)) # For each feature at index i, we keep a cumulating sum
    
    pbar = tqdm(total=output_size)
    top_idx = np.argmax(scores)
    selected = [num_list[top_idx]]
    pbar.update(1)
    for j in range(1, output_size): 
        argmax = -1
        current_max = -1
        for i,f in enumerate(num_list):
            if f not in selected:
                # Left = cumulating sum of abs corr
                # Right = abs correlation btw last_selected and f
                a = (df_scaled.get_column(selected[-1])*df_scaled.get_column(f)).mean()
                # In the rare case this calculation yields a NaN, we punish by adding 1.
                # Otherwise, proceed as usual. +1 is a punishment because
                # |corr| can be at most 1. So we are enlarging the denominator, thus reducing the score.
                cumulating_abs_corr[i] += 1 if np.isnan(a) else np.abs(a)
                denominator = cumulating_abs_corr[i] / j
                new_score = scores[i] / denominator
                if new_score > current_max:
                    current_max = new_score
                    argmax = i

        selected.append(num_list[argmax])
        pbar.update(1)

    pbar.close()
    output = pl.from_records([selected], schema=["feature"]).with_columns(
        pl.arange(1, output_size+1).alias("mrmr_rank")
    ) # Maybe unncessary to return a dataframe in this case. 
    return output.select(("mrmr_rank", "feature"))