from .type_alias import (
    PolarsFrame
    , Alternatives
    , CommonContiDist
    , SimpleDtypes
    , CPU_COUNT
    , POLARS_DATETIME_TYPES
    , POLARS_NUMERICAL_TYPES
)
from .sample import (
    lazy_sample
)
from polars.type_aliases import CorrelationMethod
from .blueprint import(
    Blueprint  # noqa: F401
)
from datetime import datetime 
from typing import Any, Optional, Tuple, Union
from itertools import combinations
from scipy.stats import (
    ks_2samp
    , kstest
)
from concurrent.futures import as_completed, ThreadPoolExecutor
from tqdm import tqdm
from math import comb
from dsds._rust import rs_levenshtein_dist
import re
import polars.selectors as cs
import polars as pl
import logging  

logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------------#
# Generic columns checks | Only works with Polars because Pandas's data types suck!            #
#----------------------------------------------------------------------------------------------#
def type_checker(df:PolarsFrame, cols:list[str], expected_type:SimpleDtypes, caller_name:str) -> bool:
    types = check_columns_types(df, cols)
    if types != expected_type:
        raise ValueError(f"The call `{caller_name}` can only be used on {expected_type} columns, not {types} types.")
    return True

def select(
    df:PolarsFrame
    , selector: Union[list[str], list[pl.Expr]]
    , persist: bool = True
) -> PolarsFrame:
    '''
    A select wrapper that makes it pipeline compatible.

    Set persist = True so that this will be remembered by the blueprint.
    '''
    if isinstance(df, pl.LazyFrame) and persist:
        return df.blueprint.select(selector)
    return df.select(selector)

def drop(df:PolarsFrame, to_drop:list[str], persist:bool=True) -> PolarsFrame:
    '''
    A pipeline compatible way to drop the given columns, which will be remembered by the blueprint
    by default.
    '''
    if isinstance(df, pl.LazyFrame) and persist:
        return df.blueprint.drop(to_drop)
    return df.drop(to_drop)

def get_numeric_cols(df:PolarsFrame, exclude:Optional[list[str]]=None) -> list[str]:
    '''Returns numerical columns except those in exclude.'''
    if exclude is None:
        selector = cs.numeric()
    else:
        selector = cs.numeric() & ~cs.by_name(exclude)
    return df.select(selector).columns

def get_string_cols(df:PolarsFrame, exclude:Optional[list[str]]=None) -> list[str]:
    '''Returns string columns except those in exclude.'''
    if exclude is None:
        selector = cs.string()
    else:
        selector = cs.string() & ~cs.by_name(exclude)
    return df.select(selector).columns

def get_datetime_cols(df:PolarsFrame) -> list[str]:
    '''Returns only datetime columns. This will not try to infer date from strings.'''
    return df.select(cs.datetime()).columns

def get_bool_cols(df:PolarsFrame) -> list[str]:
    '''Returns boolean columns.'''
    return df.select(cs.by_dtype(pl.Boolean)).columns

def get_cols_regex(df:PolarsFrame, pattern:str, lowercase:bool=False) -> list[str]:
    '''
    Returns columns that have names matching the regex pattern. If lowercase is true, match 
    using the lowercased column names.
    '''
    if lowercase:
        return (
            df.rename({c:c.lower() for c in df.columns})
            .select(cs.matches(pattern=pattern)).columns
        )
    else:
        return df.select(cs.matches(pattern=pattern)).columns


def rename(df:PolarsFrame, rename_dict:dict[str, str]) -> PolarsFrame:
    '''
    Renames the columns in df according to rename_dicts.

    This is remembered by the blueprint by default.
    '''
    d = {}
    for k, v in rename_dict.items():
        if k in df.columns:
            d[k] = v
        else:
            logger.warn(f"Attempting to rename {k}, which does not exist in df. Ignored.")

    exprs = [pl.col(c).alias(d[c]) if c in d else pl.col(c) for c in df.columns]
    return select(df, exprs)

def lowercase(df:PolarsFrame) -> PolarsFrame:
    '''
    Lowercases all column names.

    This is remembered by the blueprint by default.
    '''
    return rename(df, {c: c.lower() for c in df.columns})

# def camel_to_snake(df:PolarsFrame) -> PolarsFrame:
#     '''
#     Turn all camel case column names into snake case. This might 

#     This is remembered by the blueprint by default.
#     '''
#     pat = re.compile(r"(?<!^)(?=[A-Z])")
#     pat2 = re.compile(r"(\s|_)")
#     exprs = [pl.col(c).alias(pat.sub("_", c).lower()) if pat.search(c) and pat2.search(c) is None else pl.col(c) 
#              for c in df.columns]
#     return select(df, exprs)

def drop_nulls(
    df:PolarsFrame
    , subset:Optional[Union[list[str], str]] = None
    , persist: bool = False
) -> PolarsFrame:
    '''
    A wrapper function for Polars' drop_nulls so that it can be used in pipeline. Equivalent to
    filter by pl.all_horizontal([pl.col(c).is_null() for c in subset]).

    Set persist = True so that this will be remembered by the blueprint.
    '''
    if isinstance(df, pl.LazyFrame) and persist:
        if isinstance(subset, str):
            by = [subset]
        elif isinstance(subset, list):
            by = subset
        else:
            by = df.columns
        expr = pl.all_horizontal([pl.col(by).is_null()])
        return df.blueprint.filter(expr)
    return df.drop_nulls(subset)

def filter(
    df:PolarsFrame
    , condition: pl.Expr
    , persist: bool = False
) -> PolarsFrame:
    ''' 
    A wrapper function for Polars' filter so that it can be used in pipeline.

    Set persist = True so that this will be remembered by the blueprint.
    '''
    if isinstance(df, pl.LazyFrame) and persist:
        return df.blueprint.filter(condition)
    return df.filter(condition)

def order_by(
    df: PolarsFrame
    , by: Union[str, list[str]]
    , descending:bool = False
    , nulls_last:bool = False
    , persist: bool = False
) -> PolarsFrame:
    '''
    A wrapper function for Polars' sort so that it can be used in pipeline.

    Set persist = True so that this will be remembered by the blueprint.
    '''
    output = df.sort(by=by, descending=descending, nulls_last=nulls_last)
    if isinstance(df, pl.LazyFrame) and persist:
        return output.blueprint.add_func(df, order_by, {"by":by,"descending":descending, "nulls_last":nulls_last})
    return output

def check_binary_target(df:PolarsFrame, target:str) -> bool:
    '''
    Checks if target is binary or not. Returns true only when binary target has 0s and 1s.
    '''
    target_uniques = df.lazy().select(pl.col(target).unique()).collect()[target]
    if len(target_uniques) != 2:
        logger.error("Target is not binary.")
        return False
    elif not (0 in target_uniques and 1 in target_uniques):
        logger.error("The binary target is not encoded as 0s and 1s.")
        return False
    return True
    
def check_target_cardinality(df:PolarsFrame, target:str, raise_null:bool=True) -> pl.DataFrame:
    '''
    Returns a dataframe showing the cardinality of different target values. If raise_null = True, raise 
    an exception if target column has any null values.
    '''
    output = df.lazy().group_by(target).count().sort(target).with_columns(
        pct = pl.col("count")/pl.col("count").sum()
    ).collect()
    if raise_null and output[target].null_count() > 0:
        raise ValueError("Target contains null.")
    return output

def format_categorical_target(
    df:PolarsFrame
    , target:str
) -> Tuple[PolarsFrame, dict[Union[str, int], int]]:
    '''
    Apply an ordinal encoding to the target column for classification problems. This step helps you quickly
    turn strings into multiple categories, but is not pipeline compatbile. This returns a target-modified df
    and a mapping dict. It is recommended that you do this step outside the pipeline.

    !!! This will NOT be persisted in blueprint and does NOT work in pipeline. !!!

    Parameters
    ----------
    df
        Either a lazy or an eager Polars dataframe
    target
        Name of target column
    persist
        Wheter or not this will be persisted by the blueprint

    Example
    -------
    >>> import dsds.prescreen as ps
    ... df = pl.DataFrame({
    ...     "target":["A", "B", "C", "A", "B"]
    ... })
    ... new_df, mapping = ps.format_categorical_target(df, target="target")
    >>> print(new_df)
    >>> print(mapping)
    shape: (5, 1)
    ┌────────┐
    │ target │
    │ ---    │
    │ u16    │
    ╞════════╡
    │ 0      │
    │ 1      │
    │ 2      │
    │ 0      │
    │ 1      │
    └────────┘
    {'A': 0, 'B': 1, 'C': 2}
    '''
    uniques = df.lazy().select(
        t = pl.col(target).unique().sort()
    ).collect().drop_in_place("t")
    mapping = dict(zip(uniques, range(len(uniques))))
    output = df.with_columns(
        pl.col(target).map_dict(mapping, return_dtype=pl.UInt16)
    )
    return output, mapping

def sparse_to_dense_target(df:PolarsFrame, target:str) -> PolarsFrame:
    '''
    If target column's elements are like [0,0,1], [0,1,0], etc. for a multicategorical 
    classification problem, this will turn the target column into 2, 1, etc. This may return 
    non-sensical results if you have more than one element >= 1 in the list.

    !!! This step will NOT be remembered by the blueprint !!!

    Parameters
    ----------
    df
        Either a lazy or an eager Polars dataframe
    target
        Name of target column

    Example
    -------
    >>> import dsds.prescreen as ps
    ... df = pl.DataFrame({
    ...     "target":[[0,0,1], [0,1,0], [1,0,0], [0,1,0], [1,0,0]]
    ... })
    >>> print(ps.sparse_to_dense_target(df, target="target"))
    shape: (5, 1)
    ┌────────┐
    │ target │
    │ ---    │
    │ u32    │
    ╞════════╡
    │ 2      │
    │ 1      │
    │ 0      │
    │ 1      │
    │ 0      │
    └────────┘
    '''
    _ = type_checker(df, [target], "list", "sparse_to_dense_target")
    return df.with_columns(
        pl.col(target).list.arg_max().alias(target)
    )

def dense_to_sparse_target(df:PolarsFrame, target:str) -> PolarsFrame:
    '''
    This turns dense target column into a sparse column. This steps assume your classification target 
    is dense, meaning all categories have already been encoded to values in range 0,...,n_classes-1. If
    your target is not dense, see `dsds.prescreen.format_categorical_target`.

    !!! This step will NOT be remembered by the blueprint !!!

    Parameters
    ----------
    df
        Either a lazy or an eager Polars dataframe
    target
        Name of target column

    Example
    -------
    >>> import dsds.prescreen as ps
    ... df = pl.DataFrame({
    ...     "target":[0,1,2,1,2,0]
    ... })
    >>> print(ps.dense_to_sparse_target(df, target="target"))
    shape: (6, 1)
    ┌───────────┐
    │ target    │
    │ ---       │
    │ list[u8]  │
    ╞═══════════╡
    │ [1, 0, 0] │
    │ [0, 1, 0] │
    │ [0, 0, 1] │
    │ [0, 1, 0] │
    │ [0, 0, 1] │
    │ [1, 0, 0] │
    └───────────┘
    '''
    _ = type_checker(df, [target], "numeric", "dense_to_sparse_target")
    n_unique = df.lazy().select(
        n_unique = pl.col(target).max() + 1
    ).collect().item(0,0)
    return df.with_columns(
        pl.col(target).apply(
            lambda i: pl.zeros(n_unique, dtype=pl.UInt8, eager=True).set_at_idx(i, 1)
        )
    )

def check_columns_types(df:PolarsFrame, cols:Optional[list[str]]=None) -> str:
    '''
    Returns the unique types of given columns in a single string. If multiple types are present
    they are joined by a |. If cols is not given, automatically uses all columns.
    '''
    if cols is None:
        check_cols:list[str] = df.columns
    else:
        check_cols:list[str] = cols 

    types = sorted(set(dtype_mapping(t) for t in df.select(check_cols).dtypes))
    return "|".join(types) if len(types) > 0 else "other/unknown"

# dtype can be a "pl.datatype" or just some random data for which we want to infer a generic type.
def dtype_mapping(d: Any) -> SimpleDtypes:
    if isinstance(d, str) or d == pl.Utf8:
        return "string"
    elif isinstance(d, bool) or d == pl.Boolean:
        return "bool"
    elif isinstance(d, (int,float)) or d in POLARS_NUMERICAL_TYPES:
        return "numeric"
    elif isinstance(d, pl.List):
        return "list" # Too many possibilities. Keep it to list for now.
    elif isinstance(d, datetime) or d in POLARS_DATETIME_TYPES:
        return "datetime"
    else:
        return "other/unknown"

#----------------------------------------------------------------------------------------------#
# Prescreen Inferral, Removal Methods                                                          #
#----------------------------------------------------------------------------------------------#

def corr_table(
    df: PolarsFrame
    , features: list[str]
    , targets: list[str]
    , method:CorrelationMethod = "pearson"
) -> pl.DataFrame:
    '''
    A quick prescreen method to check the correlation between columns in cols and columns in corr_with.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    features
        Features you want to check correlation with targets
    targets
        Target columns you want to check correlation with features
    method
        Either `pearson` or `spearman`

    Example
    -------
    >>> import dsds.prescreen as ps
    ... df = pl.read_csv("../data/advertising.csv") # The dataset in data folder in the repo
    >>> print(ps.corr_table(df, features=["Age", "Daily Internet Usage"], targets=["Clicked on Ad", "Age Band"]))
    shape: (2, 3)
    ┌──────────────────────┬───────────────┬──────────┐
    │ features             ┆ Clicked on Ad ┆ Age Band │
    │ ---                  ┆ ---           ┆ ---      │
    │ str                  ┆ f64           ┆ f64      │
    ╞══════════════════════╪═══════════════╪══════════╡
    │ Age                  ┆ 0.492531      ┆ 0.950884 │
    │ Daily Internet Usage ┆ -0.786276     ┆ -0.33626 │
    └──────────────────────┴───────────────┴──────────┘
    '''
    temp = (
        df.lazy().select(
            pl.lit(f).alias("features"),
            *(pl.corr(f, t, method=method).alias(t) for t in targets)
        )
        for f in features
    )
    return pl.concat(pl.collect_all(temp))

# Add a slim option that returns fewer stats? This is generic describe.
def describe(
    df:PolarsFrame
    , sample_frac:float = 1.0
    , percentiles: list[float] = [0.25, 0.75]
    , exclude: Optional[list[str]] = None
) -> pl.DataFrame:
    '''
    A more detailed profile the data than Polars' default. This is an expensive function. Please sample 
    and exclude some columns if runtime is important.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    sample_frac
        If > 0 and < 1, will run profiling on a sample. This is recommended for bigger dataframes
    percentiles
        Percentile cuts that will be returned
    exclude
        List of columns to exclude
    '''
    selector = cs.all()
    if exclude is not None:
        selector = selector & ~cs.by_name(exclude)
        
    if sample_frac > 0 and sample_frac < 1:
        if isinstance(df, pl.LazyFrame):
            df_local = lazy_sample(df.select(selector), sample_frac=sample_frac).collect()
        else:
            df_local = df.select(selector).sample(fraction=sample_frac)
    else:
        df_local = df.lazy().select(selector).collect()

    temp = df_local.describe(percentiles)
    desc = temp.drop_in_place("describe")
    # Get unique
    unique_counts = get_unique_count(df_local).with_columns(
        unique_pct = pl.col("n_unique") / len(df_local)
    )
    # Skew and Kurtosis
    skew_and_kt_data = df_local.lazy().select(
        pl.all().skew().prefix("skew:")
        , pl.all().skew().prefix("kurtosis:")
    ).collect().row(0)

    n_cols = len(df_local.columns)
    skew_and_kt = pl.from_records((df_local.columns, skew_and_kt_data[:n_cols], skew_and_kt_data[n_cols:])
                                  , schema=["column", "skew", "kurtosis"])

    # Get a basic string description of the data type.
    dtypes_dict = dict(zip(df_local.columns, map(dtype_mapping, df_local.dtypes)))
    # Get all percentiles
    pat = re.compile("^\d+%$")
    pctls = [d for d in desc if pat.search(d)]
    # Numerical stuff
    nums = ["count" ,"null_count", "mean", "std", "median"] + pctls
    # Combine all
    final = temp.transpose(include_header=True, column_names=desc).with_columns(
        pl.col(c).cast(pl.Float64) for c in nums
    ).with_columns(
        null_pct = pl.col("null_count")/pl.col("count")
        , non_null_count = pl.col("count") - pl.col("null_count")
        , CoV = pl.col("std") / pl.col("mean")
        , dtype = pl.col("column").map_dict(dtypes_dict)
    ).join(unique_counts, on="column").join(skew_and_kt, on="column")
    # select only the stuff we need
    return final.select('column', 'dtype', "non_null_count", 'null_count','null_pct','n_unique'
                        , 'unique_pct','mean','std', 'CoV','min','max','median',"skew", "kurtosis"
                        , *pctls)

# Numeric only describe. Be more detailed.

# String only describe. Be more detailed about interesting string stats.

def describe_str_meta(
    df:PolarsFrame
    , cols: Optional[list[str]] = None
    , words_to_count:Optional[list[str]]=None
) -> pl.DataFrame:
    '''
    Computes some meta statistics about the string columns. Optionally you may pass a list
    of strings to compute the total occurrences of each of the words in the string columns. Meta
    stats inclues average, min, max, median byte length, avg count of white spaces, avg count of 
    digits, average count of capitalized letters, average count of lowercase letters and word counts 

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    cols
        If not provided, will compute for all string columns
    words_to_count
        If provided, word counts will be computed
    '''
    if cols is None:
        strs = get_string_cols(df)
    else:
        _ = type_checker(df, cols, "string", "describe_str_meta")
        strs = cols
        
    df_str = df.lazy().select(strs)
    nstrs = len(strs)
    stats = df_str.select(
        pl.all().null_count().prefix("nc:"),
        pl.all().max().prefix("max:"),
        pl.all().min().prefix("min:"),
        pl.all().mode().first().prefix("mode:"),
        pl.all().str.lengths().min().prefix("min_byte_len:"),
        pl.all().str.lengths().max().prefix("max_byte_len:"),
        pl.all().str.lengths().mean().prefix("avg_byte_len:"),
        pl.all().str.lengths().median().prefix("median_byte_len:"),
        pl.all().str.count_match(r"\s").mean().prefix("avg_space_cnt:"),
        pl.all().str.count_match(r"[0-9]").mean().prefix("avg_digit_cnt:"),
        pl.all().str.count_match(r"[A-Z]").mean().prefix("avg_cap_cnt:"),
        pl.all().str.count_match(r"[a-z]").mean().prefix("avg_lower_cnt:")
    ).collect().row(0)
    output = {
        "features":strs,
        "null_count": stats[:nstrs],
        "min": stats[nstrs: 2*nstrs],
        "max": stats[2*nstrs: 3*nstrs],
        "mode": stats[3*nstrs: 4*nstrs],
        "min_byte_len": stats[4*nstrs: 5*nstrs],
        "max_byte_len": stats[5*nstrs: 6*nstrs],
        "avg_byte_len": stats[6*nstrs: 7*nstrs],
        "median_byte_len": stats[7*nstrs: 8*nstrs],
        "avg_space_cnt": stats[8*nstrs: 9*nstrs],
        "avg_digit_cnt": stats[9*nstrs: 10*nstrs],
        "avg_cap_cnt": stats[10*nstrs: 11*nstrs],
        "avg_lower_cnt": stats[11*nstrs: ],
    }

    if isinstance(words_to_count, list):
        for w in words_to_count:
            output["total_"+ w + "_count"] = df_str.select(
                                                pl.all().str.count_match(w).sum().prefix("wc:")
                                            ).collect().row(0)

    return pl.from_dict(output)

def describe_categories(
    df:PolarsFrame
    , cols: Optional[list[str]] = None
) -> pl.DataFrame:
    '''
    Describes the string categories by looking at the count for each category. Some stats include min/max/avg 
    successive difference in term of count for the categories.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    cols
        If not provided, will compute for all string columns
    '''
    if cols is None:
        strs = get_string_cols(df)
    else:
        _ = type_checker(df, cols, "string", "describe_categories")
        strs = cols

    stats = (
        df.lazy().group_by(s).agg(
            pl.count()
        ).sort(by=[pl.col("count"), pl.col(s)]).select(
            pl.lit(s).alias("feature"),
            pl.col(s).count().alias("n_unique"),
            pl.when(pl.col(s).null_count() > 0).then(True).otherwise(False).alias("has_null"),
            pl.col(s).first().alias("category_w_min_count"),
            pl.col("count").min().alias("min_count"),
            pl.col(s).last().alias("category_w_max_count"),
            pl.col("count").max().alias("max_count"),
            pl.col("count").mean().alias("avg_count"),
            pl.col("count").diff().min().alias("min_successive_diff"),
            pl.col("count").diff().mean().alias("avg_successive_diff"),
            pl.col("count").diff().max().alias("max_successive_diff"),
        )
        for s in strs
    )
    return pl.concat(pl.collect_all(stats))

# Add an outlier description

# -----------------------------------------------------------------------------------------------
def non_numeric_removal(df:PolarsFrame, include_bools:bool=False) -> PolarsFrame:
    '''
    Removes all non-numeric columns. If include_bools = True, then keep boolean columns. This will 
    be remembered by blueprint by default.
    '''
    if include_bools:
        selector = ~(cs.numeric()|cs.by_dtype(pl.Boolean))
    else:
        selector = ~cs.numeric()

    non_nums = df.select(selector).columns
    logger.info(f"The following columns are dropped because they are not numeric: {non_nums}.\n"
                f"Removed a total of {len(non_nums)} columns.")
    
    return drop(df, non_nums)

# Check if columns are duplicates. Might take time.
def duplicate_inferral():
    # Get profiles first.
    # Divide into categories: bools, strings, numerics, datetimes.
    # Then cut down list to columns that have the same min, max, n_unique and null_count.
    # Then check equality..
    pass

def infer_by_pattern(
    df: PolarsFrame
    , pattern:str
    , sample_frac:float = 0.75
    , sample_count:int = 100_000
    , sample_rounds:int = 3
    , threshold:float = 0.9
    , count_null:bool = False
) -> list[str]:
    '''
    Find all string columns whose elements reasonably match the given pattern. The match logic can 
    be tuned using the all the parameters.

    Parameters
    ----------
    sample_frac
        The pct of the total dataframe to use as pool
    sample_count
        From the pool, how many rows to sample for each round 
    sample_rounds
        How many rounds of sampling we are doing
    threshold
        For each round, what is the match% that is needed to be a counted as a success. For instance, 
        in round 1, for column x, we have 92% match rate, and threshold = 0.9. We count column x as a match for 
        this round. In the end, the column must match for every round to be considered a real match.
    count_null
        For individual matches, do we want to count null as a match or not? If the column has high null pct,
        the non-null values might mostly match the pattern. In this case, using count_null = True will match the column, 
        while count_null = False will most likely exclude the column.

    Returns:
        a list of columns that pass the matching test
    
    '''
    strs = get_string_cols(df)
    df_local = lazy_sample(df.lazy(), sample_frac=sample_frac).collect()    
    matches:set[str] = set(strs)
    sample_size = min(sample_count, len(df_local)-1)
    for _ in range(sample_rounds):
        df_sample = df_local.sample(n = sample_size)
        fail = df_sample.select(
            (
                pl.when(pl.col(s).is_null()).then(count_null).otherwise(
                    pl.col(s).str.contains(pattern)
                ).sum()/sample_size
            ).alias(s) 
            for s in strs
        ).transpose(include_header=True, column_names=["pattern_match_pct"])\
        .filter(pl.col("pattern_match_pct") < threshold)["column"]
        # If the match failes in this round, remove the column.
        matches.difference_update(fail)

    return list(matches)

def remove_by_pattern(
    df: PolarsFrame
    , pattern:str
    , sample_pct:float = 0.75
    , sample_count:int = 100_000
    , sample_rounds:int = 3
    , threshold:float = 0.9
    , count_null:bool = False
) -> PolarsFrame:
    '''
    Calls infer_by_pattern and removes those columns that are inferred. This will be remembered by blueprint by default.
    '''
    
    remove_cols = infer_by_pattern(
        df
        , pattern
        , sample_pct
        , sample_count
        , sample_rounds
        , threshold 
        , count_null
    )
    logger.info(f"The following columns are dropped because they match the element pattern: {pattern}.\n"
                f"{remove_cols}\n"
                f"Removed a total of {len(remove_cols)} columns.")
    
    return drop(df, remove_cols)

def infer_emails(
    df: PolarsFrame
    , sample_pct:float = 0.75
    , sample_count:int = 100_000
    , sample_rounds:int = 3
    , threshold:float = 0.9
    , count_null:bool = False
) -> list[str]:
    # Why does this regex not work?
    # r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+'
    return infer_by_pattern(
        df
        , r'\S+@\S+\.\S+'
        , sample_pct
        , sample_count
        , sample_rounds
        , threshold 
        , count_null
    )

def remove_emails(
    df: PolarsFrame
    , sample_pct:float = 0.75
    , sample_count:int = 100_000
    , sample_rounds:int = 3
    , threshold:float = 0.9
    , count_null:bool = False
) -> PolarsFrame:
    '''
    Calls infer_emails and removes those columns that are inferred. This will be remembered by blueprint by default.
    '''
    
    emails = infer_emails(df, sample_pct, sample_count, sample_rounds, threshold, count_null)
    logger.info(f"The following columns are dropped because they are emails. {emails}.\n"
            f"Removed a total of {len(emails)} columns.")
    
    return drop(df, emails)

# Check for columns that are US zip codes.
# Might add options for other countries later.
def infer_zipcodes():
    # Match string using pattern inferral
    # Take a look at integers too, are they always 5 digits? 
    pass

def infer_dates(df:PolarsFrame) -> list[str]:
    '''Infers date columns in dataframe. This inferral is not perfect.'''
    logger.info("Date Inferral is error prone due to the huge variety of date formats. Please use with caution.")
    dates = df.select(cs.datetime()).columns
    strings = get_string_cols(df)
    # MIGHT REWRITE THIS LOGIC
    # Might be memory intensive on big dataframes.
    sample_size = min(len(df), 100_000)
    sample_df = df.lazy().select(strings)\
        .drop_nulls().collect()\
        .sample(n = sample_size).select(
            # Cleaning the string first. Only try to catch string dates which are in the first split by space
           pl.col(s).str.strip().str.replace_all("(/|\.)", "-").str.split(by=" ").list.first() 
           for s in strings
        )
    for s in strings:
        try:
            c = sample_df[s].str.to_date(strict=False)
            if 1 - c.null_count()/sample_size >= 0.15: # if at least 15% valid (able to be converted)
                # This last check is to account for single digit months.
                # 3/3/1995 will not be parsed to a string because standard formats require 03/03/1995
                # At least 15% of dates naturally have both month and day as 2 digits numbers
                dates.append(s)
        except: # noqa: E722
            # Very stupid code, but I have to do it...
            pass
    
    return dates

def remove_dates(df:PolarsFrame) -> PolarsFrame:
    '''
    Removes all date columns from dataframe. This algorithm will try to infer if string column is date. 
    This will be remembered by blueprint by default.
    '''

    remove_cols = infer_dates(df) 
    logger.info(f"The following columns are dropped because they are dates. {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    return drop(df, remove_cols)

def infer_invalid_numeric(df:PolarsFrame, threshold:float=0.5, include_null:bool=False) -> list[str]:
    '''
    Infers numeric columns that have more than threshold pct of invalid (NaN) values. Inf and -Inf values
    are considered as NaN.
    
    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    threshold
        Columns with higher than threshold null pct will be dropped. Threshold should be between 0 and 1.
    include_null
        If true, then null values will also be counted as invalid.
    '''
    nums = get_numeric_cols(df)
    df_local = df.lazy().select(nums).with_row_count(offset=1).set_sorted("row_nr")
    if include_null:
        expr = ((pl.col(nums).is_nan())|(pl.col(nums).is_infinite())|(pl.col(nums).is_null())).sum()/pl.col("row_nr").max()
    else:
        expr = ((pl.col(nums).is_nan())|(pl.col(nums).is_infinite())).sum()/pl.col("row_nr").max()
    
    temp = df_local.select(expr).collect()
    return [c for c, pct in zip(temp.columns, temp.row(0)) if pct >= threshold]

def infer_nan(df: PolarsFrame) -> list[str]:
    '''
    Returns all columns that contain NaN or infinite values.
    '''
    nums = get_numeric_cols(df)
    nan_counts = df.lazy().select(((pl.col(nums).is_nan()) | (pl.col(nums).is_infinite())).sum()).collect().row(0)
    return [col for col, cnt in zip(nums, nan_counts) if cnt > 0]

def remove_invalid_numeric(df:PolarsFrame, threshold:float=0.5, include_null:bool=False) -> PolarsFrame:
    '''
    Removes numeric columns that have more than threshold pct of invalid (NaN) values. 
    
    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    threshold
        Columns with higher than threshold null pct will be dropped. Threshold should be between 0 and 1.
    include_null
        If true, then null values will also be counted as invalid.
    '''
    remove_cols = infer_invalid_numeric(df, threshold, include_null) 
    logger.info(f"The following columns are dropped because they have more than {threshold*100:.2f}%"
                f" invalid values. {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    return drop(df, remove_cols)

def infer_nulls(df:PolarsFrame, threshold:float=0.5) -> list[str]:
    '''
    Infers columns that have more than threshold pct of null values. Use infer_invalid_numeric with 
    include_null = True if you want to find high NaN + Null columns.
    
    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    threshold
        Columns with higher than threshold null pct will be dropped. Threshold should be between 0 and 1.
    '''
    temp = df.lazy().with_row_count(offset=1).select(
        pl.all().exclude("row_nr").null_count() / pl.col("row_nr").max()
    ).collect()
    return [c for c, pct in zip(temp.columns, temp.row(0)) if pct >= threshold]

def remove_nulls(df:PolarsFrame, threshold:float=0.5) -> PolarsFrame:
    '''
    Removes columns with more than threshold pct of null values. Use remove_invalid_numeric with 
    include_null = True if you want to drop columns with high (NaN + Null)%.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    threshold
        Columns with higher than threshold null pct will be dropped. Threshold should be between 0 and 1.
    '''
    remove_cols = infer_nulls(df, threshold) 
    logger.info(f"The following columns are dropped because they have more than {threshold*100:.2f}%"
                f" null values. {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    return drop(df, remove_cols)

def infer_by_var(df:PolarsFrame, threshold:float, target:str) -> list[str]:
    '''Infers columns that have lower than threshold variance. Target will not be included.'''
    nums = get_numeric_cols(df, exclude=[target])
    vars = df.lazy().select(pl.col(nums).var()).collect()
    return [c for c, v in zip(vars.columns, vars.row(0)) if v < threshold]

def remove_by_var(df:PolarsFrame, threshold:float, target:str) -> PolarsFrame:
    '''
    Removes features with low variance. Features with > threshold variance will be kept. 
    Threshold should be positive. This will be remembered by blueprint by default.
    '''

    remove_cols = infer_by_var(df, threshold, target) 
    logger.info(f"The following columns are dropped because they have lower than {threshold} variance. {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    return drop(df, remove_cols)

# Really this is just an alias
infer_by_regex = get_cols_regex

def remove_by_regex(df:PolarsFrame, pattern:str, lowercase:bool=False) -> PolarsFrame:
    '''
    Remove columns if their names satisfy the given regex rules. This is common when you want to remove columns 
    with certain prefixes that may not be allowed to use in models.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    pattern
        The regex pattern
    lowercase
        Whether to lowercase everything and then match
    '''
    remove_cols = get_cols_regex(df, pattern, lowercase)
    logger.info(f"The following columns are dropped because their names satisfy the regex rule: {pattern}."
                f" {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    
    return drop(df, remove_cols)

def get_unique_count(
    df:PolarsFrame, 
    include_null_count:bool=False,
    estimate:bool=False) -> pl.DataFrame:
    '''
    Gets unique counts for columns and returns a dataframe with schema = ["column", "n_unique"]. Null count
    is useful in knowing if null is one of the unique values and thus is included as an option. Note that
    null != NaN.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    include_null_count
        If true, this will return a dataframe with schema = ["column", "n_unique", "null_count"]
    estimate
        If true, use HyperLogLog algorithm to estimate n_unique
    '''
    if estimate:
        n_unique = pl.all().approx_n_unique()
    else:
        n_unique = pl.all().n_unique()
    
    if include_null_count:
        temp = df.lazy().select(
            n_unique,
            pl.all().null_count().suffix("_null_count")
        ).collect().row(0)
        n = len(df.columns)
        return pl.from_records((df.columns, temp[:n], temp[n:]), schema=["column", "n_unique", "null_count"])
    else:
        return df.lazy().select(
            n_unique
        ).collect().transpose(include_header=True, column_names=["n_unique"])

def infer_highly_unique(df:PolarsFrame, threshold:float=0.9, estimate:bool=False) -> list[str]:
    '''
    Infers columns that have higher than threshold unique pct.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    threshold
        Every column with unique pct higher than this threshold will be returned.
    estimate
        If true, use HyperLogLog algorithm to estimate n_uniques. This is only recommended
        when dataframe is absolutely large.
    '''
    if estimate:
        unique_pct = pl.all().exclude("row_nr").approx_n_unique() / pl.col("row_nr").max()
    else:
        unique_pct = pl.all().exclude("row_nr").n_unique() / pl.col("row_nr").max()
    temp = df.lazy().with_row_count(offset=1).set_sorted("row_nr").select(
        unique_pct
    ).collect()
    return [c for c, pct in zip(temp.columns, temp.row(0)) if pct > threshold]

def remove_by_uniqueness(df:PolarsFrame, threshold:float=0.9) -> PolarsFrame:
    '''
    Remove columns that have higher than threshold pct of unique values. Usually this is done to filter
    out id-like columns.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or an eager Polars dataframe
    threshold
        The threshold for unique pct. Columns with higher than this threshold unique pct will be removed 
    '''
    remove_cols = infer_highly_unique(df, threshold)
    logger.info(f"The following columns are dropped because more than {threshold*100:.2f}% of unique values."
                f" {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    return drop(df, remove_cols)

# Once there is a config, add a discrete criterion config
def infer_discretes(df:PolarsFrame
    , threshold:float=0.1
    , max_n_unique:int=100
    , exclude:Optional[list[str]]=None
) -> list[str]:
    '''
    A column that satisfies either n_unique < max_n_unique or unique_pct < threshold 
    will be considered discrete. E.g. threshold = 0.1 and max_n_unique = 100 means if a 
    column has < 100 unique values, or the unique pct is < 10%, then it will be considered
    as discrete.

    Parameters
    ----------
    df
        Either a lazy or an eager Polars dataframe
    threshold
        The threshold for unique pct
    max_n_unique
        The maximum number of unique values allowed for a column to be considered discrete
    exclude
        List of columns to exclude
    '''
    exclude_list = [] if exclude is None else exclude
    exclude_list.append("row_nr")
    temp = get_unique_count(df.with_row_count(offset=1).set_sorted("row_nr"))
    len_df = temp.filter(pl.col("column") == "row_nr").item(0,1)
    return temp.filter(
        ((pl.col("n_unique") < max_n_unique) | (pl.col("n_unique") < len_df * threshold)) 
        & (~pl.col("column").is_in(exclude_list)) # is not in
    )["column"].to_list()

def infer_conti(
    df:PolarsFrame
    , discrete_threshold:float = 0.1
    , discrete_max_n_unique:int = 100
    , exclude:Optional[list[str]]=None
) -> list[str]:
    '''
    Returns everything that is not considered discrete.

    Parameters
    ----------
    df
        Either a lazy or an eager Polars dataframe
    discrete_threshold
        The threshold for unique pct in discrete inferral
    discrete_max_n_unique
        The maximum number of unique values allowed for a column to be considered discrete
    exclude
        List of columns to exclude
    '''
    exclude_list = infer_discretes(df, discrete_threshold, discrete_max_n_unique)
    if exclude is not None:
        exclude_list.extend(exclude)
    return df.select(cs.numeric() & ~cs.by_name(exclude_list)).columns

def infer_constants(df:PolarsFrame, include_null:bool=True) -> list[str]:
    '''
    Returns a list of inferred constant columns.
    
    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    include_null
        If true, then columns with two distinct values like [value_1, null] will be considered a 
        constant column
    '''
    condition = (pl.col("n_unique") == 1)
    if include_null:
        condition = condition | ((pl.col("n_unique") == 2) & (pl.col("null_count") > 0))

    return get_unique_count(df, True).filter(condition)["column"].to_list()

def infer_binary(df:PolarsFrame, include_null:bool=True) -> list[str]:
    '''
    Returns a list of inferred binary columns. (Columns with 2 values).
    
    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    include_null
        If true, then columns with three distinct values like ['Y', 'N', null] will be considered a 
        binary column
    '''
    condition = (pl.col("n_unique") == 2)
    if include_null:
        condition = condition | ((pl.col("n_unique") == 3) & (pl.col("null_count") > 0))

    return get_unique_count(df, True).filter(condition)["column"].to_list()

def infer_n_unique(df:PolarsFrame, n:int, include_null:bool=True, leq:bool=False) -> list[str]:
    ''' 
    Returns a list of columns with exactly n unique values. If leq = True, this returns a list
    of columns with <= n unique values.
    
    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    n
        The number of distinct values
    include_null
        If true, this will return columns with n+1 distinct values where one of them is null.
    leq
        If true, this will return columns with <= n unique values
    '''
    if leq:
        condition1 = (pl.col("n_unique").le(n))
        condition2 = (pl.col("n_unique").le(n+1))
    else:
        condition1 = (pl.col("n_unique").eq(n))
        condition2 = (pl.col("n_unique").eq(n+1))

    if include_null:
        condition = condition1 | (condition2 & (pl.col("null_count") > 0))

    return get_unique_count(df, True).filter(condition)["column"].to_list()

def get_complement(
    df: PolarsFrame,
    cols: list[str]
) -> list[str]:
    '''
    A convenience method that returns all columns in df but not in cols.
    '''
    return [c for c in df.columns if c not in cols]

def get_similar_names(
    df: PolarsFrame,
    ref: str,
    distance:int = 3
) -> list[str]:
    '''
    Returns columns whose name is within `distance` Levenshtein distance from the reference string. This 
    is useful when there is a typo in certain column names, e.g. "count" vs. "coutn". 
    '''
    return [c for c in df.columns if rs_levenshtein_dist(c, ref) <= distance]

def remove_constants(df:PolarsFrame, include_null:bool=True) -> PolarsFrame:
    '''
    Removes all constant columns from dataframe. This will be remembered by blueprint by default.
    
    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    include_null
        If true, then columns with two distinct values like [value_1, null] will be considered a 
        constant column
    '''
    remove_cols = infer_constants(df, include_null)
    logger.info(f"The following columns are dropped because they are constants. {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    return drop(df, remove_cols)

def infer_nums_from_str(df:PolarsFrame, ignore_comma:bool=True) -> list[str]:
    '''
    Infers hidden numeric columns which are stored as strings like "$5.55" or "#123". If 
    ignore_comma = True, then it will first filter out all "," in the string.
    '''
    expr = pl.col(df.select(cs.string()).columns)
    if ignore_comma:
        expr = expr.str.replace_all(",", "")

    temp = df.lazy().select(
        expr.str.count_match("\d*\.?\d+").mean()
    ).collect()
    # On average, about 1 occurrence of number in the values.
    return [c for c, avg in zip(temp.columns, temp.row(0)) if avg <= 1 and avg >= 0.95]

def infer_coordinates(df:PolarsFrame):
    pass

def infer_numlist_from_str(df:PolarsFrame):
    pass

def remove_if_exists(df:PolarsFrame, cols:list[str]) -> PolarsFrame:
    '''Removes the given columns if they exist in the dataframe. This will be remembered by blueprint by default.'''
    remove_cols = list(set(cols).intersection(df.columns))
    logger.info(f"The following columns are dropped. {remove_cols}.\nRemoved a total of {len(remove_cols)} columns.")
    return drop(df, remove_cols)

def estimate_n_unique(
    df: PolarsFrame,
    cols: list[str]
) -> pl.DataFrame:
    '''
    Applies the HyperLogLog algorithm to estimate unique count. This is only recommended for absolutely 
    large dataframes.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    cols
        Columns to estimate n_uniques
    '''
    return df.lazy().select(
        pl.col(cols).approx_n_unique().suffix("_n_unique_est")
    ).collect().transpose(include_header=True, column_names=["estimated_n_uniques"])

def shrink_dtype(
    df:PolarsFrame,
    cols: Optional[list[str]] = None
) -> PolarsFrame:
    '''
    A pipeline compatible shrink dtype for numeric columns to help lower pressure on memory. The shrinked dtype
    will be the smallest possible to hold all values in the columns. Note that for really big dataframes, 
    it is still better to preemptively specify the dtypes in the scan/read statement.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    cols
        List of numeric columns which we want to shrink. If provided, they must all be numeric columns. 
        If not, will infer all numeric columns.
    '''
    if cols is None:
        to_shrink = get_numeric_cols(df)
    else:
        _ = type_checker(df, cols, "numeric", "shrink_dtype")
        to_shrink = cols

    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(pl.col(to_shrink).shrink_dtype())
    return df.with_columns(pl.col(to_shrink).shrink_dtype())

#----------------------------------------------------------------------------------------------#
# More statistical Methods
#----------------------------------------------------------------------------------------------#

def _ks_compare(
    df:pl.DataFrame
    , pair:Tuple[str, str]
    , alt:Alternatives="two-sided"
) -> Tuple[Tuple[str, str], float, float]:
    res = ks_2samp(df[pair[0]], df[pair[1]], alt)
    return (pair, res.statistic, res.pvalue)

def ks_compare(
    df:PolarsFrame
    , test_cols:Optional[list[str]] = None
    , alt: Alternatives = "two-sided"
    , smaple_frac:float = 0.75
    , skip:int = 0
    , max_comp:int = 1000
) -> pl.DataFrame:
    '''
    Run ks-stats on all non-discrete columns in the dataframe. If test_cols is None, it will infer
    continuous columns. See docstring of discrete_inferral to see what is considered discrete. Continuous is 
    considered to be non-discrete. Since ks 2 sample comparison is relatively expensive, we will
    always sample 75% of the dataset, unless the user specifies a different sample_frac. This function will
    collect lazy frames at the sample_fraction rate.

    Note: this will only run on all 2 combinations of columns, starting from skip and end at skip + max_comp.

    Note: The null hypothesis is that the two columns come from the same distribution. Therefore a small p-value means
    that they do not come from the same distribution. Having p-value > threshold does not mean they have the same 
    distribution automatically, and it requires more examination to reach the conclusion.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    cols
        List of numeric columns which we want to test. If none, will use all non-discrete numeric columns.
    alt
        The alternative for the statistic test. One of `two-sided`, `greater` or `less`
    sample_frac
        Sampling fraction to run the test.
    skip
        The number of comparisons to skip
    max_comp
        The total number of comparisons to make in this run. A strict upper bound is len(test_cols) choose 2.
    '''
    if test_cols is None:
        nums = [f for f in get_numeric_cols(df) if f not in infer_discretes(df)]
    else:
        _ = type_checker(df, test_cols, "numeric", "ks_compare")
        nums = test_cols

    nums.sort()
    if isinstance(df, pl.LazyFrame):
        df_test = lazy_sample(df.select(nums).lazy(), sample_frac=smaple_frac).collect()
    else:
        df_test = df.select(nums).sample(fraction=smaple_frac)

    n_c2 = comb(len(nums), 2)
    last = min(skip + max_comp, n_c2)
    results = []
    to_test = enumerate(combinations(nums, 2))
    pbar = tqdm(total=min(max_comp, n_c2 - skip), desc="Comparisons")
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as ex:
        for f in as_completed(ex.submit(_ks_compare, df_test, pair, alt) 
                              for i, pair in to_test if i < last and i > skip):
            results.append(f.result())
            pbar.update(1)

    pbar.close()
    return pl.from_records(results, schema=["combination", "ks-stats", "p-value"])

def _dist_inferral(df:pl.DataFrame, c:str, dist:CommonContiDist) -> Tuple[str, float, float]:
    res = kstest(df[c], dist)
    return (c, res.statistic, res.pvalue)

def dist_test(
    df: PolarsFrame
    , which_dist:CommonContiDist
    , smaple_frac:float = 0.75
) -> pl.DataFrame:
    '''
    Tests if the numeric columns follow the given distribution by using the KS test. The null 
    hypothesis is that the columns follow the given distribution. We sample 75% of data because 
    ks test is relatively expensive.
    '''
    nums = get_numeric_cols(df)
    if isinstance(df, pl.LazyFrame):
        df_test = lazy_sample(df.select(nums).lazy(), sample_frac=smaple_frac).collect()
    else:
        df_test = df.select(nums).sample(fraction=smaple_frac)

    results = []
    pbar = tqdm(total=len(nums), desc="Comparisons")
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as ex:
        for f in as_completed(ex.submit(_dist_inferral, df_test, c, which_dist) for c in nums):
            results.append(f.result())
            pbar.update(1)

    pbar.close()
    return pl.from_records(results, schema=["feature", "ks-stats", "p_value"])

def suggest_normal(
    df:PolarsFrame

    , threshold:float = 0.05
) -> list[str]:
    '''
    Suggests which columns are normally distributed. This takes the columns for which the null hypothesis
    cannot be rejected in the dist_test (KS test).
    '''
    return dist_test(df, "norm").filter(pl.col("p_value") > threshold)["feature"].to_list()

def suggest_uniform(
    df:PolarsFrame
    , threshold:float = 0.05
) -> list[str]:
    '''
    Suggests which columns are uniformly distributed. This takes the columns for which the null hypothesis
    cannot be rejected in the dist_test (KS test).
    '''
    return dist_test(df, "uniform").filter(pl.col("p_value") > threshold)["feature"].to_list()

def suggest_lognormal(
    df:PolarsFrame
    , threshold:float = 0.05
) -> list[str]:
    '''
    Suggests which columns are log-normally distributed. This takes the columns which the null hypothesis
    cannot be rejected in the dist_test (KS test).
    '''
    return dist_test(df, "lognorm").filter(pl.col("p_value") > threshold)["feature"].to_list()

def suggest_dist(
    df:PolarsFrame
    , threshold:float = 0.05
    , dist: CommonContiDist = "norm"
) -> list[str]:
    '''
    Suggests which columns follow the given distribution. This returns the columns which the null hypothesis
    cannot be rejected in the dist_test (KS test).
    '''
    return dist_test(df, dist).filter(pl.col("p_value") > threshold)["feature"].to_list()