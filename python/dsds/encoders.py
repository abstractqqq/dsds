from __future__ import annotations

from .type_alias import (
    PolarsFrame
)
from .prescreen import (
    get_bool_cols
    , get_string_cols
    , get_unique_count
    , check_binary_target
    , type_checker
)
from .blueprint import( # Need this for Polars extension to work
    Blueprint  # noqa: F401
)
from typing import Optional, Union, Any
import numpy as np
import polars as pl
import logging

logger = logging.getLogger(__name__)

def boolean_encode(df:PolarsFrame, keep_null:bool=True) -> PolarsFrame:
    '''
    Converts all boolean columns into binary 0, 1 columns.

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
        temp = df.lazy().select(str_cols).select(
            pl.all().unique().implode().list.sort()
        )
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
            )["column"].to_list()
    else:
        binary_list = cols
    
    return one_hot_encode(df, cols=binary_list, drop_first=True, separator=separator)

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
    binary_list = get_unique_count(df).filter(pl.col("n_unique") == 2)["column"]
    temp = df.lazy().select(binary_list).select(
            pl.all().unique().implode().list.sort()
        )
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

def multicat_one_hot_encode(
    df:PolarsFrame
    , cols: list[str]
    , delimiter: str = "|"
    , drop_first: bool = False
) -> PolarsFrame:
    '''
    Expands multicategorical columns into several one-hot-encoded columns respectively. A multicategorical column is a 
    column with strings like `aaa|bbb|ccc`, which means this row belongs to categories aaa, bbb, and ccc. Typically, 
    such a column will contain strings separated by a delimiter. This method will collect all unique strings separated 
    by the delimiter and one hot encode the corresponding column, e.g. by checking if `aaa` is contained in values of this
    column. Nulls will be mapped to 0 in the generated one-hot columns. If you wish to have a null mask, take a look 
    at `dsds.encoders.missing_indicator`.

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
        If true, the first category in the each column will be dropped.

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
    temp = df.lazy().select(cols).select(
        pl.all().str.split("|").explode().unique().implode().list.sort()
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
                for i in range(start_index, len(u)) if isinstance(u[i], str)
            )
        else:
            logger.info(f"The multicategorical column {c.name} seems to have only 1 unique value. Dropped.")

    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(exprs).blueprint.drop(cols)
    return df.with_columns(exprs).drop(cols)


def ordinal_auto_encode(
    df:PolarsFrame
    , cols:Optional[list[str]]=None
    , descending:bool = False
    , exclude:Optional[list[str]]=None
) -> PolarsFrame:
    '''
    Automatically applies ordinal encoding to the provided columns by the order of the elements. This method is 
    great for string columns like age ranges, with values like ["10-20", "20-30"], etc. (Beware of string lengths,
    e.g. if "100-110" exists in age range, then it may mess up the natural order.)

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

    temp = df.lazy().select(ordinal_list).select(
        pl.all().unique().implode().list.sort(descending=descending)
    )
    for t in temp.collect().get_columns():
        uniques:pl.Series = t[0]
        mapping = {t.name: uniques, "to": list(range(len(uniques)))}
        if isinstance(df, pl.LazyFrame):
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
    , default:Optional[int] = None
) -> PolarsFrame:
    '''
    Ordinal encode the columns in the ordinal_mapping dictionary.

    This will be remembered by blueprint by default.
        
    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    ordinal_mapping
        A dictionary that looks like {"a":{"a1":1, "a2":2}, ...}
    default
        Default value for values not mentioned in the ordinal_mapping dict.
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
    , min_samples_leaf:int = 20
    , smoothing:float = 10.
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
    p = df.lazy().select(pl.col(target).mean()).collect().item(0,0)
    is_lazy = isinstance(df, pl.LazyFrame)
    # If c has null, null will become a group when we group by.
    for c in str_cols:
        ref = df.groupby(c).agg(
            pl.count().alias("cnt"),
            pl.col(target).mean().alias("cond_p")
        ).with_columns(
            (1./(1. + ((-(pl.col("cnt").cast(pl.Float64) - min_samples_leaf))/smoothing).exp())).alias("alpha")
        ).select(
            pl.col(c),
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
    , mapping: Union[dict[str, dict[Any, Any]], list[pl.Expr] , pl.Expr]
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
            raise ValueError("Target is not binary or not properly encoded or contains nulls.")

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