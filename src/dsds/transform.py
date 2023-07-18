from __future__ import annotations

from .type_alias import (
    PolarsFrame
    , ImputationStrategy
    , ScalingStrategy
    , PowerTransformStrategy
    , DateExtract
    , clean_strategy_str
    , CPU_COUNT
)
from .prescreen import (
    get_bool_cols
    , get_string_cols
    , get_unique_count
    , check_binary_target
    , check_columns_types
    , type_checker
)
from .blueprint import( # Need this for Polars extension to work
    Blueprint
)
import logging
import math
import numpy as np
import polars as pl
from typing import Optional, Tuple, Any
from scipy.stats._morestats import (
    yeojohnson_normmax
    , boxcox_normmax
)
from concurrent.futures import as_completed, ThreadPoolExecutor
from tqdm import tqdm

# A lot of companies are still using Python < 3.10
# So I am not using match statements

logger = logging.getLogger(__name__)

def impute(
    df:PolarsFrame
    , cols:list[str]
    , strategy:ImputationStrategy = 'median'
    , const:float = 1.
) -> PolarsFrame:
    '''
    Impute the given columns with the given strategy.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        The columns to impute
    strategy
        One of 'median', 'mean', 'const' or 'mode'. If 'const', the const argument should be provided. Note that
        if strategy is mode and if two values occur the same number of times, a random one will be picked.
    const
        The constant value to impute by if strategy = 'const'    
    '''
    s = clean_strategy_str(strategy)
    if s == "median":
        all_medians = df.lazy().select(cols).median().collect().row(0)
        exprs = (pl.col(c).fill_null(all_medians[i]) for i,c in enumerate(cols))
    elif s in ("mean", "avg", "average"):
        all_means = df.lazy().select(cols).mean().collect().row(0)
        exprs = (pl.col(c).fill_null(all_means[i]) for i,c in enumerate(cols))
    elif s in ("const", "constant"):
        exprs = (pl.col(c).fill_null(const) for c in cols)
    elif s in ("mode", "most_frequent"):
        all_modes = df.lazy().select(cols).select(pl.all().mode().first()).collect().row(0)
        exprs = (pl.col(c).fill_null(all_modes[i]) for i,c in enumerate(cols))
    else:
        raise TypeError(f"Unknown imputation strategy: {strategy}")

    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(list(exprs))
    return df.with_columns(exprs)

def scale(
    df:PolarsFrame
    , cols:list[str]
    , strategy:ScalingStrategy="standard"
    , const:float = 1.0
) -> PolarsFrame:
    '''
    Scale the given columns with the given strategy.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        The columns to scale
    strategy
        One of 'standard', 'min_max', 'const'. If 'const', the const argument should be provided
    const
        The constant value to scale by if strategy = 'const'    
    '''
    _ = type_checker(df, cols, "numeric", "scale")
    s = clean_strategy_str(strategy)
    if s == "standard":
        mean_std = df.lazy().select(cols).select(
            pl.all().mean().prefix("mean:")
            , pl.all().std().prefix("std:")
        ).collect().row(0)
        exprs = ( (pl.col(c) - mean_std[i])/(mean_std[i + len(cols)]) for i,c in enumerate(cols) )
    elif s == "min_max":
        min_max = df.lazy().select(cols).select(
            pl.all().min().prefix("min:"),
            pl.all().max().prefix("max:")
        ).collect().row(0) # All mins come first, then maxs
        exprs = ( (pl.col(c) - min_max[i])/((min_max[i + len(cols)] - min_max[i])) for i,c in enumerate(cols) )
    elif s in ("const", "constant"):
        exprs = (pl.col(c)/const for c in cols)
    else:
        raise TypeError(f"Unknown scaling strategy: {strategy}")

    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(list(exprs))
    return df.with_columns(exprs)

def boolean_transform(df:PolarsFrame, keep_null:bool=True) -> PolarsFrame:
    '''
    Converts all boolean columns into binary columns.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    keep_null
        If true, null will be kept. If false, null will be mapped to 0.
    '''
    bool_cols = get_bool_cols(df)
    if keep_null: # Directly cast. If null, then cast will also return null
        exprs = (pl.col(c).cast(pl.UInt8) for c in bool_cols)
    else: # Cast. Then fill null to 0s.
        exprs = (pl.col(c).cast(pl.UInt8).fill_null(0) for c in bool_cols)

    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(exprs)
    return df.with_columns(exprs)

def missing_indicator(
    df: PolarsFrame
    , cols: Optional[list[str]] = None
    , suffix: str = "_missing"
) -> PolarsFrame:
    '''
    Add one-hot columns for missing values in the given columns.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        If not provided, will create missing indicators for all columns
    suffix
        The suffix given to the missing indicator columns
    '''
    if cols is None:
        to_add = df.columns
    else:
        to_add = cols
    one = pl.lit(1, dtype=pl.UInt8)
    zero = pl.lit(0, dtype=pl.UInt8)
    exprs = (pl.when(pl.col(c).is_null()).then(one).otherwise(zero).suffix(suffix) for c in to_add)
    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(list(exprs))
    return df.with_columns(exprs)

def merge_infreq_values(
    df: PolarsFrame
    , cols: list[str]
    , min_count: int | None = 10
    , min_frac: float | None = None
    , separator: str = '|'
) -> PolarsFrame:
    '''
    Combines infrequent categories in string columns together.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        List of string columns to perform this operation
    min_count
        Define a category to be infrequent if it occurs less than min_count. This defaults to 10 if both min_count and 
        min_frac are None.
    min_frac
        Define category to be infrequent if it occurs less than this percentage of times. If both min_count and min_frac
        are set, min_frac takes priority
    separator
        The separator for the new value representing the combined categories

    Example
    -------
    >>> import dsds.transform as t
    ... df = pl.DataFrame({
    ...     "a":["a", "b", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c"],
    ...     "b":["a", "b", "c", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d"]
    ... })
    >>> df
    shape: (14, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ str ┆ str │
    ╞═════╪═════╡
    │ a   ┆ a   │
    │ b   ┆ b   │
    │ c   ┆ c   │
    │ c   ┆ d   │
    │ …   ┆ …   │
    │ c   ┆ d   │
    │ c   ┆ d   │
    │ c   ┆ d   │
    │ c   ┆ d   │
    └─────┴─────┘
    >>> t.merge_infreq_values(df, ["a", "b"], min_count=3)
    shape: (14, 2)
    ┌─────┬───────┐
    │ a   ┆ b     │
    │ --- ┆ ---   │
    │ str ┆ str   │
    ╞═════╪═══════╡
    │ a|b ┆ a|c|b │
    │ a|b ┆ a|c|b │
    │ c   ┆ a|c|b │
    │ c   ┆ d     │
    │ …   ┆ …     │
    │ c   ┆ d     │
    │ c   ┆ d     │
    │ c   ┆ d     │
    │ c   ┆ d     │
    └─────┴───────┘
    '''
    _ = type_checker(df, cols, "string", "merge_infreq_values")
    if min_frac is None:
        if min_count is None:
            comp = pl.col("count") < 10
        else:
            comp = pl.col("count") < min_count
    else:
        comp = pl.col("count")/pl.col("count").sum() < min_frac

    exprs = []
    for c in cols:
        infreq = df.lazy().groupby(c).count().filter(
            comp
        ).collect().get_column(c)
        value = separator.join(infreq)
        exprs.append(
            pl.when(pl.col(c).is_in(infreq)).then(value).otherwise(pl.col(c)).alias(c)
        )
    
    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(exprs)
    return df.with_columns(exprs)

def one_hot_encode(
    df:PolarsFrame
    , cols:Optional[list[str]]=None
    , separator:str="_"
    , drop_first:bool=False
) -> PolarsFrame:
    '''
    One-hot-encode the given columns.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        If not provided, will use all string columns
    separator
        The separator used in the names of the new columns
    drop_first
        If true, the first category in the each column will be dropped. E.g. if column "D" has 3 distinct values, 
        say 'A', 'B', 'C', then only two binary indicators 'D_B' and 'D_C' will be created. This is useful for
        reducing dimensions and also good for optimization methods that require data to be non-degenerate.
    '''
    
    if isinstance(cols, list):
        _ = type_checker(df, cols, "string", "one_hot_encode")
        str_cols = cols
    else:
        str_cols = get_string_cols(df)

    if isinstance(df, pl.LazyFrame):
        temp = df.lazy().select(str_cols).groupby(1).agg(
            pl.all().unique().sort()
        ).select(str_cols)
        exprs:list[pl.Expr] = []
        start_index = int(drop_first)
        one = pl.lit(1, dtype=pl.UInt8) # Avoid casting 
        zero = pl.lit(0, dtype=pl.UInt8) # Avoid casting
        for t in temp.collect().get_columns():
            u:pl.List = t[0] # t is a Series which contains a single series/list, so u is a series/list
            if len(u) > 1:
                exprs.extend(
                    pl.when(pl.col(t.name) == u[i]).then(one).otherwise(zero).alias(t.name + separator + u[i])
                    for i in range(start_index, len(u))
                )
            else:
                logger.info(f"During one-hot-encoding, the column {t.name} is found to have 1 unique value. Dropped.")
        
        return df.blueprint.with_columns(exprs).blueprint.drop(str_cols)
    else:
        return df.to_dummies(columns=str_cols, separator=separator, drop_first=drop_first)
    
def binary_encode(
    df:PolarsFrame
    , cols:Optional[list[str]]=None
    , separator: str = "_"
    , exclude:Optional[list[str]]=None
) -> PolarsFrame:
    '''
    Encode binary string columns as 0s and 1s depending on the order of the 2 unique strings. E.g. if the two unique 
    values are 'N' and 'Y', then 'N' will be mapped to 0 and 'Y' to 1 because 'N' < 'Y'. This is essentially 
    one-hot-encode for binary string columns with drop_first = True.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        If not provided, will use all string columns
    separator
        The separator used in the names of the new columns
    '''

    if cols is None:
        str_cols = get_string_cols(df)
        exclude = [] if exclude is None else exclude
        binary_list = get_unique_count(df)\
            .filter( # Binary + Not Exclude + Only String
                (pl.col("n_unique") == 2) & (~pl.col("column").is_in(exclude)) & (pl.col("column").is_in(str_cols))
            ).get_column("column").to_list()
    else:
        binary_list = cols
    
    return one_hot_encode(df, cols=binary_list, drop_first=True, separator=separator)

def multicat_one_hot_encode(
    df:PolarsFrame
    , cols: list[str]
    , delimiter: str
    , drop_first: bool = False
) -> PolarsFrame:
    '''
    Expands multicategorical columns into several one-hot-encoded columns respectively. A multicategorical column is a 
    column with strings like `aaa|bbb|ccc`, where it means this row belongs to categories aaa, bbb, and ccc. Typically, 
    such a column will contain strings separated by a delimiter. This method will collect all unique strings separated 
    by the delimiter and one hot encode the corresponding column.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        If not provided, will use all string columns
    separator
        The separator used in the names of the new columns
    drop_first
        If true, the first category in the each column will be dropped. E.g. if column "D" has 3 distinct values, 
        say 'A', 'B', 'C', then only two binary indicators 'D_B' and 'D_C' will be created. This is useful for
        reducing dimensions and also good for optimization methods that require data to be non-degenerate.

    Returns
    -------
        A lazy/eager dataframe with multicategorical columns one-hot-encoded

    Example
    -------
    >>> df = pl.DataFrame({
    ... "text1":["abc|ggg", "abc|sss", "ccc|abc"],
    ... "text2":["aaa|bbb", "ccc|aaa", "bbb|ccc"]
    ... })
    >>> df
    shape: (3, 2)
    ┌─────────┬─────────┐
    │ text1   ┆ text2   │
    │ ---     ┆ ---     │
    │ str     ┆ str     │
    ╞═════════╪═════════╡
    │ abc|ggg ┆ aaa|bbb │
    │ abc|sss ┆ ccc|aaa │
    │ ccc|abc ┆ bbb|ccc │
    └─────────┴─────────┘
    >>> multicat_one_hot_encode(df, cols=["text1", "text2"], delimiter="|")
    shape: (3, 7)
    ┌───────────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
    │ text1|abc ┆ text1|ccc ┆ text1|ggg ┆ text1|sss ┆ text2|aaa ┆ text2|bbb ┆ text2|ccc │
    │ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       │
    │ u8        ┆ u8        ┆ u8        ┆ u8        ┆ u8        ┆ u8        ┆ u8        │
    ╞═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╡
    │ 1         ┆ 0         ┆ 1         ┆ 0         ┆ 1         ┆ 1         ┆ 0         │
    │ 1         ┆ 0         ┆ 0         ┆ 1         ┆ 1         ┆ 0         ┆ 1         │
    │ 1         ┆ 1         ┆ 0         ┆ 0         ┆ 0         ┆ 1         ┆ 1         │
    └───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
    '''
    _ = type_checker(df, cols, "string", "multicat_one_hot_encode")
    temp = df.lazy().select(cols).groupby(1).agg(
        pl.all().str.split(delimiter).explode().unique().sort()
    ).select(cols) 
    one = pl.lit(1, dtype=pl.UInt8) # Avoid casting 
    zero = pl.lit(0, dtype=pl.UInt8) # Avoid casting
    exprs = []
    start_index = int(drop_first)
    for c in temp.collect().get_columns():
        u = c[0]
        if len(u) > 1:
            exprs.extend(
                pl.when(pl.col(c.name).str.contains(u[i])).then(one).otherwise(zero).alias(c.name + delimiter + u[i])
                for i in range(start_index, len(u))
            )
        else:
            logger.info(f"The multicategorical column {c.name} seems to have only 1 unique value. Dropped.")

    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(exprs).blueprint.drop(cols)
    return df.with_columns(exprs).drop(cols)

def force_binary(df:PolarsFrame) -> PolarsFrame:
    '''
    Force every binary column, no matter what data type, to be turned into 0s and 1s according to the order of the 
    elements. If a column has two unique values like [null, "haha"], then null will be mapped to 0 and "haha" to 1.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    '''
    binary_list = get_unique_count(df).filter(pl.col("n_unique") == 2).get_column("column")
    temp = df.lazy().select(binary_list).groupby(1).agg(
            pl.all().unique().sort()
        ).select(binary_list)
    exprs:list[pl.Expr] = []
    one = pl.lit(1, dtype=pl.UInt8) # Avoid casting 
    zero = pl.lit(0, dtype=pl.UInt8) # Avoid casting
    for t in temp.collect().get_columns():
        u:pl.List = t[0] # t is a Series which contains a single list which contains the 2 unique values 
        exprs.append(
            pl.when(pl.col(t.name) == u[0]).then(zero).otherwise(one).alias(t.name)
        )

    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(exprs)
    return df.with_columns(exprs)

def ordinal_auto_encode(
    df:PolarsFrame
    , cols:Optional[list[str]]=None
    , descending:bool = False
    , exclude:Optional[list[str]]=None
) -> PolarsFrame:
    '''
    Automatically applies ordinal encoding to the provided columns by the order of the elements. This method is 
    great for string columns like age ranges, with values like ["10-20", "20-30"], etc.

    This will be remembered by blueprint by default.
        
    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        If not provided, will use all string columns
    descending
        If true, will use descending order (0 will be mapped to largest element)
    exclude
        Columns to exclude. This is only used when cols is not provided.
    '''
    if isinstance(cols, list):
        _ = type_checker(df, cols, "string", "ordinal_auto_encode")
        ordinal_list = cols
    else:
        ordinal_list = get_string_cols(df, exclude=exclude)

    temp = df.lazy().groupby(1).agg(
        pl.col(c).unique().sort(descending=descending) for c in ordinal_list
    ).select(ordinal_list)
    for t in temp.collect().get_columns():
        uniques:pl.Series = t[0]
        mapping = {t.name: uniques, "to": list(range(len(uniques)))} 
        if isinstance(df, pl.LazyFrame):
            # Use a list here because Python cannot pickle a generator
            df = df.blueprint.map_dict(t.name, mapping, "to", None)
        else:
            map_tb = pl.DataFrame(mapping)
            df = df.join(map_tb, on = t.name).with_columns(
                pl.col("to").alias(t.name)
            ).drop("to")
    return df

def ordinal_encode(
    df:PolarsFrame
    , ordinal_mapping:dict[str, dict[str,int]]
    , default:int|None=None
) -> PolarsFrame:
    '''
    Ordinal encode the columns in the ordinal_mapping dictionary. The ordinal_mapping dict should look like:
    {"a":{"a1":1, "a2":2}, ...}, which means for column a, a1 should be mapped to 1, a2 mapped to 2. Values 
    not mentioned in the dict will be mapped to default.

    This will be remembered by blueprint by default.
        
    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    ordinal_mapping
        A dictionary that looks like {"a":{"a1":1, "a2":2}, ...}
    default
        Default value for values not mentioned in the dict.
    '''

    for c in ordinal_mapping:
        if c in df.columns:
            mapping = ordinal_mapping[c]
            if isinstance(df, pl.LazyFrame):
                # This relies on the fact that dicts in Python is ordered
                mapping = {c: mapping.keys(), "to": mapping.values()}
                df = df.blueprint.map_dict(c, mapping, "to", default)
            else:
                mapping = pl.DataFrame((mapping.keys(), mapping.values()), schema=[c, "to"])
                df = df.join(mapping, on = c, how="left").with_columns(
                    pl.col("to").fill_null(default).alias(c)
                ).drop("to")
        else:
            logger.warning(f"Found that column {c} is not in df. Skipped.")

    return df

def smooth_target_encode(
    df:PolarsFrame
    , target:str
    , cols:list[str]
    , min_samples_leaf:int
    , smoothing:float
    , check_binary:bool=True
) -> PolarsFrame:
    '''
    Smooth target encoding for binary classification. Currently only implemented for binary target.

    This will be remembered by blueprint by default.
    
    See https://towardsdatascience.com/dealing-with-categorical-variables-by-using-target-encoder-a0f1733a4c69

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    target
        Name of the target column
    cols
        If not provided, will use all string columns
    min_samples_leaf
        The k in the smoothing factor equation
    smoothing
        The f of the smoothing factor equation 
    check_binary
        Checks if target is binary. If not, throw an error
    '''
    if isinstance(cols, list):
        _ = type_checker(df, cols, "string", "smooth_target_encode")
        str_cols = cols
    else:
        str_cols = get_string_cols(df)
    
    # Only works for binary target for now. There is a non-binary ver of target encode, but I
    # am just delaying the implementation...
    if check_binary:
        if not check_binary_target(df, target):
            raise ValueError("Target is not binary or not properly encoded.")

    # probability of target = 1
    p = df.lazy().select(pl.col(target).mean()).collect().row(0)[0]
    is_lazy = isinstance(df, pl.LazyFrame)
    # If c has null, null will become a group when we group by.
    for c in str_cols:
        ref = df.groupby(c).agg(
            pl.count().alias("cnt"),
            pl.col(target).mean().alias("cond_p")
        ).with_columns(
            (1./(1. + ((-(pl.col("cnt").cast(pl.Float64) - min_samples_leaf))/smoothing).exp())).alias("alpha")
        ).select(
            pl.col(c).alias(c),
            to = pl.col("alpha") * pl.col("cond_p") + (pl.lit(1) - pl.col("alpha")) * pl.lit(p)
        ) # If df is lazy, ref is lazy. If df is eager, ref is eager
        if is_lazy:
            df = df.blueprint.map_dict(c, ref.collect().to_dict(), "to", None)
        else: # It is ok to do inner join because all values of c are present in ref.
            df = df.join(ref, on = c).with_columns(
                pl.col("to").alias(c)
            ).drop("to")
    return df

def _when_then_repl(c:str, repl_map:dict):
    expr = pl.col(c)
    for og, repl in repl_map.items():
        expr = pl.when(pl.col(c).eq(og)).then(repl).otherwise(expr)
    
    return expr.alias(c)

def feature_mapping(
    df:PolarsFrame
    , mapping: dict[str, dict[Any, Any]] | list[pl.Expr] | pl.Expr
) -> PolarsFrame:
    '''
    Maps specific values of a feature into values provided. This is a common task when the feature columns come with 
    error codes.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    mapping
        Either a dict like {"a": {999: None, 998: None, 997: None}, ...}, meaning that 999, 998 and 997 in column "a" 
        should be replaced by null, or a list/a single Polars (when-then) expression(s) like the following,  
        pl.when(pl.col("a") >= 997).then(None).otherwise(pl.col("a")).alias("a"), which will perform the same mapping 
        as the dict example. Note that using Polars expression can tackle more complex replacement.

    Example
    -------
    >>> df = pl.DataFrame({
    ...     "a": [1,2,3,998,999],
    ...     "b": [999, 1,2,3,4]
    ... })
    >>> print(df)
    shape: (5, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 999 │
    │ 2   ┆ 1   │
    │ 3   ┆ 2   │
    │ 998 ┆ 3   │
    │ 999 ┆ 4   │
    └─────┴─────┘
    >>> feature_mapping(df, mapping = {"a":{998:None,999:None}, "b":{999:None}})
    shape: (5, 2)
    ┌──────┬──────┐
    │ a    ┆ b    │
    │ ---  ┆ ---  │
    │ i64  ┆ i64  │
    ╞══════╪══════╡
    │ 1    ┆ null │
    │ 2    ┆ 1    │
    │ 3    ┆ 2    │
    │ null ┆ 3    │
    │ null ┆ 4    │
    └──────┴──────┘
    >>> mapping = [pl.when(pl.col("a")>=998).then(None).otherwise(pl.col("a")).alias("a")
    ...          , pl.when(pl.col("b")==999).then(None).otherwise(pl.col("b")).alias("b")]
    >>> feature_mapping(df, mapping)
    shape: (5, 2)
    ┌──────┬──────┐
    │ a    ┆ b    │
    │ ---  ┆ ---  │
    │ i64  ┆ i64  │
    ╞══════╪══════╡
    │ 1    ┆ null │
    │ 2    ┆ 1    │
    │ 3    ┆ 2    │
    │ null ┆ 3    │
    │ null ┆ 4    │
    └──────┴──────┘
    '''
    if isinstance(mapping, dict):
        exprs = []
        for c, repl_map in mapping.items():
            exprs.append(_when_then_repl(c, repl_map))
    elif isinstance(mapping, list):
        exprs = []
        for f in mapping:
            if isinstance(f, pl.Expr):
                exprs.append(f)
            else:
                logger.warn(f"Found {f} is not a Polars expression. Ignored.")
    elif isinstance(mapping, pl.Expr):
        exprs = [mapping]
    else:
        raise TypeError("The argument `mapping` must be one of the following types: "
                        "dict[str, dict[Any, Any]] | list[pl.Expr] | pl.Expr")
    
    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(exprs)
    return df.with_columns(exprs)

def custom_binning(
    df:PolarsFrame
    , cols:list[str]
    , cuts:list[float]
    , suffix:str = ""
) -> PolarsFrame:
    '''
    Bins according to the cuts provided. The same cuts will be applied to all columns in cols.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        Numerical columns that will be binned
    cuts
        A list of floats representing break points in the intervals
    suffix
        If you don't want to replace the original columns, you have the option to give the binned column a suffix
    '''
    if isinstance(df, pl.LazyFrame):
        exprs = [
            pl.col(c).cut(cuts).cast(pl.Utf8).suffix(suffix) for c in cols
        ]
        return df.blueprint.with_columns(exprs)
    else:
        return df.with_columns(
            pl.col(c).cut(cuts).cast(pl.Utf8).suffix(suffix) for c in cols
        )
    
def fixed_sized_binning(
    df:PolarsFrame
    , cols:list[str]
    , interval: float
    , suffix:str = ""
) -> PolarsFrame:
    '''
    Bins according to fixed interval size. The same cuts will be applied to all columns in cols. Bin will 
    start from min(feature) to max(feature) + interval with step length = interval.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        Numerical columns that will be binned
    interval
        The fixed sized interval
    suffix
        If you don't want to replace the original columns, you have the option to give the binned column a suffix
    '''
    bounds = df.lazy().select(cols).select(
        pl.all().min().prefix("min:")
        , pl.all().max().prefix("max:")
    ).collect().row(0)
    exprs = []
    n = len(cols)
    for i, c in enumerate(cols):
        cut = np.arange(bounds[i], bounds[n+i] + interval, step=interval).tolist()
        exprs.append(pl.col(c).cut(cut).cast(pl.Utf8).suffix(suffix))

    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(exprs)
    return df.with_columns(exprs)

def quantile_binning(
    df:PolarsFrame
    , cols:list[str]
    , n_bins:int
    , suffix:str = ""
) -> PolarsFrame:
    '''
    Bin a continuous variable into categories, based on quantile. Null values will be its own category. The same binning
    rule will be applied to all columns in cols. If you want different n_bins for different columns, chain another 
    quantile_binning with different cols and n_bins.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    cols
        A list of numeric columns. This has to be supplied by the user because it is not recommended
        to bin all numerical variables
    n_bins
        The number of desired bins. If n_bins = 4, the quantile cuts will be [0.25,0.5,0.74], and 4 
        categories will be created, which represent values ranging from (-inf, 0.25 quantile value],
        (0.25 quantile value, 0.5 quantile value],...(0.75 quantile value, inf]
    suffix
        If you don't want to replace the original columns, you have the option to give the binned column a suffix

    Example
    -------
    >>> df = pl.DataFrame({
    ...     "a":range(5)
    ... })
    >>> df
    shape: (5, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 0   │
    │ 1   │
    │ 2   │
    │ 3   │
    │ 4   │
    └─────┘
    >>> quantile_binning(df, cols=["a"], n_bins=4)
    shape: (5, 1)
    ┌───────────┐
    │ a         │
    │ ---       │
    │ str       │
    ╞═══════════╡
    │ (-inf, 1] │
    │ (-inf, 1] │
    │ (1, 2]    │
    │ (2, 3]    │
    │ (3, inf]  │
    └───────────┘
    '''
    _ = type_checker(df, cols, "numeric", "quantile_binning")
    qcuts = np.arange(start=1/n_bins, stop=1.0, step = 1/n_bins)
    if isinstance(df, pl.LazyFrame):
        cuts = df.select(cols).select(
            pl.all().qcut(qcuts).unique().cast(pl.Utf8).str.extract(r"\((.*?),")
            .cast(pl.Float64).sort().tail(len(qcuts))
        ).collect()
        exprs = [
            pl.col(c).cut(cuts.drop_in_place(c).to_list()).cast(pl.Utf8).suffix(suffix) for c in cols
        ]
        return df.blueprint.with_columns(exprs)
    else: # Eager frame
        return df.with_columns(
            pl.col(c).qcut(qcuts).cast(pl.Utf8).suffix(suffix) for c in cols 
        )

def woe_cat_encode(
    df:PolarsFrame
    , target:str
    , cols:Optional[list[str]]=None
    , min_count:float = 1.
    , default: float = -10.
    , check_binary:bool = True
) -> PolarsFrame:
    '''
    Performs WOE encoding for categorical features. To WOE encode numerical columns, first bin them using
    custom_binning or quantile_binning. This only works for binary target.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    target
        The name of the target column
    cols
        If not provided, all string columns will be used
    min_count
        A numerical factor that prevents values like infinity to occur when taking log
    default
        Null values will be mapped to default
    check_binary
        Whether to check target is binary or not.
    '''
    if isinstance(cols, list):
        _ = type_checker(df, cols, "string", "woe_cat_encode")
        str_cols = cols
    else:
        str_cols = get_string_cols(df)

    if check_binary:
        if not check_binary_target(df, target):
            raise ValueError("Target is not binary or not properly encoded.")

    is_lazy = isinstance(df, pl.LazyFrame)
    for s in str_cols:
        ref = df.lazy().groupby(s).agg(
            ev = pl.col(target).sum()
            , nonev = (pl.lit(1) - pl.col(target)).sum()
        ).with_columns(
            ev_rate = (pl.col("ev") + min_count)/(pl.col("ev").sum() + 2.0*min_count)
            , nonev_rate = (pl.col("nonev") + min_count)/(pl.col("nonev").sum() + 2.0*min_count)
        ).with_columns(
            woe = (pl.col("ev_rate")/pl.col("nonev_rate")).log()
        ).select(
            pl.col(s)
            , pl.col("woe")
        ).collect()
        if is_lazy:
            df = df.blueprint.map_dict(s, ref.to_dict(), "woe", default)
        else:
            df = df.join(ref, on = s, how="left").with_columns(
                pl.col("woe").fill_null(default).alias(s)
            ).drop("woe")

    return df

def _lmax_estimate_step(df:PolarsFrame, c:str, s:PowerTransformStrategy) -> Tuple[str, float]:
    np_col = df.lazy().select(pl.col(c).cast(pl.Float64)).collect().get_column(c).view()
    if s in ("yeo_johnson", "yeojohnson"):
        lmax:float = yeojohnson_normmax(np_col)
    else:
        lmax:float = boxcox_normmax(np_col, method="mle")
    
    return (c, lmax)

def power_transform(
    df: PolarsFrame
    , cols: list[str]
    , strategy: PowerTransformStrategy = "yeo_johnson"
    , n_threads:int = CPU_COUNT
    # , lmbda: Optional[float] = None
) -> PolarsFrame:
    '''
    Performs power transform on the numerical columns.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    cols
        Must be explicitly provided and must all be numerical
    strategy
        Either 'yeo_johnson' or 'box_cox'
    n_threads
        The max number of worker threads to use in Python
    '''
    _ = type_checker(df, cols, "numeric", "power_transform")
    s = clean_strategy_str(strategy)
    exprs:list[pl.Expr] = []
    # Ensure columns do not have missing values
    exclude_columns_w_nulls = df.lazy().select(cols).null_count().collect().transpose(
        include_header=True, column_names=["null_count"]
    ).filter(pl.col("null_count") > 0).get_column("column").to_list()

    if len(exclude_columns_w_nulls) > 0:
        logger.info("The following columns will not be processed by power_transform because they contain missing "
                    f"values. Please impute them:\n{exclude_columns_w_nulls}")
        
    non_null_list = [c for c in cols if c not in exclude_columns_w_nulls]
    pbar = tqdm(non_null_list, desc = "Inferring best paramters")
    if s in ("yeo_johnson", "yeojohnson"):
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            for future in as_completed(ex.submit(_lmax_estimate_step, df, c, s) for c in non_null_list):
                c, lmax = future.result()
                if lmax == 0: # log(x + 1)
                    x_ge_0_sub_expr = (pl.col(c).add(1)).log()
                else: # ((x + 1)**lmbda - 1) / lmbda
                    x_ge_0_sub_expr = ((pl.col(c).add(1)).pow(lmax) - 1) / lmax

                if lmax == 2: # -log(-x + 1)
                    x_lt_0_sub_expr = pl.lit(-1) * (1 - pl.col(c)).log()
                else: #  -((-x + 1)**(2 - lmbda) - 1) / (2 - lmbda)
                    t = 2 - lmax
                    x_lt_0_sub_expr = pl.lit(-1/t) * ((1 - pl.col(c)).pow(t) - 1)

                exprs.append(
                    pl.when(pl.col(c).ge(0)).then(x_ge_0_sub_expr).otherwise(x_lt_0_sub_expr).alias(c)
                )
                pbar.update(1)

    elif s in ("box_cox", "boxcox"):
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            for future in as_completed(ex.submit(_lmax_estimate_step, df, c, s) for c in non_null_list):
                c, lmax = future.result()
                if lmax == 0: # log(x)
                    exprs.append(pl.col(c).log())
                else: # (x**lmbda - 1) / lmbda
                    exprs.append(
                        (pl.col(c).pow(lmax) - 1) / lmax
                    )
                pbar.update(1)
    else:
        raise TypeError(f"The input strategy {strategy} is not a valid strategy. Valid strategies are: yeo_johnson "
                        "or box_cox")
    pbar.close()
    if isinstance(df, pl.LazyFrame):
        return df.lazy().blueprint.with_columns(exprs)
    return df.with_columns(exprs)


# Should feature engineering spin off to its own module? Or stay in transform?
# First, I need to wait until I have more feature engineering stuff..

def normalize(
    df: PolarsFrame
    , cols:list[str]
) -> PolarsFrame:
    '''
    Normalize the given columns by dividing them with the respective column sum.

    !!!Note this will not be remember by the pipeline!!!

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    cols
        Must be explicitly provided and should all be numeric columns
    '''
    
    types = check_columns_types(df, cols)
    if types != "numeric":
        raise ValueError(f"normalize can only be used on numeric columns, not {types} types.")

    return df.with_columns(pl.col(c)/pl.col(c).sum() for c in cols)

def log_transform(
    df: PolarsFrame
    , cols:list[str]
    , base:float = math.e
    , cast_non_positive: None | float = None
) -> PolarsFrame:
    '''
    Performs classical log transform on the given columns

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    cols
        Must be explicitly provided and should all be numeric columns
    base
        Base of log. Default is math.e
    cast_non_positive
        How to deal with non positive values (<=0). None means turn them into null
    '''
    _ = type_checker(df, cols, "numeric", "log_transform")
    exprs = [
        pl.when(pl.col(c) <= 0).then(cast_non_positive).otherwise(pl.col(c).log(base)).suffix("_log") for c in cols
    ]
    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(exprs)
    return df.with_columns(exprs)

def extract_dt_features(
    df: PolarsFrame
    , cols: list[str]
    , extract: DateExtract | list[DateExtract] = ["year", "quarter", "month"]
    , sunday_first: bool = False
) -> PolarsFrame:
    '''
    Extracts additional date related features from existing date/datetime columns.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    cols
        Must be explicitly provided and should all be date/datetime columns
    extract
        One of "year", "quarter", "month", "week", "day_of_week", "day_of_year", or a list of these values
        such as ["year", "quarter"], which means extract year and quarter from all the columns provided 
    sunday_first
        For day_of_week, by default, Monday maps to 1, and so on. If sunday_first = True, then Sunday will be
        mapped to 1 and so on

    Example
    -------
    >>> import dsds.transform as t
    ... df = pl.DataFrame({
    ...     "date1":["2021-01-01", "2022-02-03", "2023-11-23"]
    ...     , "date2":["2021-01-01", "2022-02-03", "2023-11-23"]
    ... }).with_columns(
    ...     pl.col(c).str.to_date() for c in ["date1", "date2"]
    ... )
    >>> print(df)
    shape: (3, 2)
    ┌────────────┬────────────┐
    │ date1      ┆ date2      │
    │ ---        ┆ ---        │
    │ date       ┆ date       │
    ╞════════════╪════════════╡
    │ 2021-01-01 ┆ 2021-01-01 │
    │ 2022-02-03 ┆ 2022-02-03 │
    │ 2023-11-23 ┆ 2023-11-23 │
    └────────────┴────────────┘
    >>> cols = ["date1", "date2"]
    >>> print(t.extract_dt_features(df, cols=cols))
    shape: (3, 8)
    ┌────────────┬────────────┬────────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
    │ date1      ┆ date2      ┆ date1_year ┆ date2_yea ┆ date1_qua ┆ date2_qua ┆ date1_mon ┆ date2_mon │
    │ ---        ┆ ---        ┆ ---        ┆ r         ┆ rter      ┆ rter      ┆ th        ┆ th        │
    │ date       ┆ date       ┆ u16        ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       │
    │            ┆            ┆            ┆ u16       ┆ u8        ┆ u8        ┆ u8        ┆ u8        │
    ╞════════════╪════════════╪════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╡
    │ 2021-01-01 ┆ 2021-01-01 ┆ 2021       ┆ 2021      ┆ 1         ┆ 1         ┆ 1         ┆ 1         │
    │ 2022-02-03 ┆ 2022-02-03 ┆ 2022       ┆ 2022      ┆ 1         ┆ 1         ┆ 2         ┆ 2         │
    │ 2023-11-23 ┆ 2023-11-23 ┆ 2023       ┆ 2023      ┆ 4         ┆ 4         ┆ 11        ┆ 11        │
    └────────────┴────────────┴────────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
    '''
    _ = type_checker(df, cols, "datetime", "extract_dt_features")
    exprs = []
    if isinstance(extract, list):
        to_extract = extract
    else:
        to_extract = [extract]
    
    for e in to_extract:
        if e == "month":
            exprs.extend(pl.col(c).dt.month().cast(pl.UInt8).suffix("_month") for c in cols)
        elif e == "year":
            exprs.extend(pl.col(c).dt.year().cast(pl.UInt16).suffix("_year") for c in cols)
        elif e == "quarter":
            exprs.extend(pl.col(c).dt.quarter().cast(pl.UInt8).suffix("_quarter") for c in cols)
        elif e == "week":
            exprs.extend(pl.col(c).dt.week().cast(pl.UInt8).suffix("_week") for c in cols)
        elif e == "day_of_week":
            if sunday_first:
                exprs.extend(((pl.col(c).dt.weekday()+1)%7).cast(pl.UInt8).suffix("_day_of_week") for c in cols)
            else:
                exprs.extend(pl.col(c).dt.weekday().cast(pl.UInt8).suffix("_day_of_week") for c in cols)
        elif e == "day_of_year":
            exprs.extend(pl.col(c).dt.ordinal_day().cast(pl.UInt8).suffix("_day_of_year") for c in cols)
        else:
            logger.error(f"Found {e} in extract, but is not a valid DateExtract value. Ignored.")

    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(exprs)
    return df.with_columns(exprs)