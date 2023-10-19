from __future__ import annotations

from .type_alias import (
    PolarsFrame
)
from .prescreen import (
    get_string_cols
    , get_unique_count
    , check_binary_target
    , infer_multicategorical
    , type_checker
)
from .blueprint import (
    _dsds_with_columns, 
    _dsds_with_columns_and_drop,
    _dsds_map_dict
)
from typing import Optional, Union
import numpy as np
import polars as pl
import logging

logger = logging.getLogger(__name__)


def missing_indicator(
    df: PolarsFrame
    , cols: Optional[list[str]] = None
    , include_nan:bool = False
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
    include_nan
        If true, NaN and Inf values will be treated as missing
    suffix
        The suffix given to the missing indicator columns
    '''
    to_add = df.columns if cols is None else cols
    if include_nan:
        exprs = [
            pl.any_horizontal(
                pl.col(c).is_nan(),
                pl.col(c).is_infinite(),
                pl.col(c).is_null()
            ).cast(pl.UInt8).suffix(suffix) 
            for c in to_add
        ]
    else:
        exprs = [pl.col(to_add).is_null().cast(pl.UInt8).suffix(suffix)]

    return _dsds_with_columns(df, exprs)

def one_hot_encode(
    df:PolarsFrame
    , cols:Optional[list[str]]=None
    , separator:str="_"
    , drop_first:bool=False
) -> PolarsFrame:
    '''
    One-hot-encode the given columns. Null will always be dropped. If a one-hot-encoded null indicator
    is desired, please see dsds.encoders.missing_indicator.

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

    temp = df.lazy().select(
        pl.col(str_cols).unique().drop_nulls().implode().list.sort()
    )
    exprs:list[pl.Expr] = []
    start_index = int(drop_first)
    for t in temp.collect().get_columns():
        u:pl.Series = t[0] # t is a Series which contains a single series, so u is a series
        if len(u) > 1:
            exprs.extend(
                pl.col(t.name).eq(u[i]).fill_null(False).cast(pl.UInt8).alias(t.name + separator + u[i])
                for i in range(start_index, len(u))
            )
        else:
            logger.info(f"During one-hot-encoding, the column {t.name} is found to have 1 unique value. Dropped.")
    
    return _dsds_with_columns_and_drop(df, exprs, str_cols)

    
def reverse_one_hot_encode(
    df: PolarsFrame
    , root_col_name: Union[str, list[str]]
    , separator:str = "_"
) -> PolarsFrame:
    '''
    Reverses one-hot-encoded columns.

    This will be remembered by blueprint by default.

    Paramters
    ---------
    df
        Either a lazy or eager Polars DataFrame
    root_col_name
        Either the root name of a single column, or a list of root names
    separator
        The separator used to separate root name and values in the one-hot-encoded columns

    Example
    -------
    >>> print(test)
    shape: (4, 5)
    ┌─────┬─────┬─────┬─────┬─────┐
    │ a_a ┆ a_b ┆ a_c ┆ b_n ┆ b_y │
    │ --- ┆ --- ┆ --- ┆ --- ┆ --- │
    │ u8  ┆ u8  ┆ u8  ┆ u8  ┆ u8  │
    ╞═════╪═════╪═════╪═════╪═════╡
    │ 1   ┆ 0   ┆ 0   ┆ 0   ┆ 1   │
    │ 0   ┆ 1   ┆ 0   ┆ 1   ┆ 0   │
    │ 0   ┆ 0   ┆ 1   ┆ 0   ┆ 1   │
    │ 1   ┆ 0   ┆ 0   ┆ 1   ┆ 0   │
    └─────┴─────┴─────┴─────┴─────┘
    >>> enc.reverse_one_hot_encode(test, root_col_name=["a", "b"])
    shape: (4, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ str ┆ str │
    ╞═════╪═════╡
    │ a   ┆ y   │
    │ b   ┆ n   │
    │ c   ┆ y   │
    │ a   ┆ n   │
    └─────┴─────┘
    '''
    
    if isinstance(root_col_name, list):
        all_roots = root_col_name
    else:
        all_roots = [root_col_name]

    exprs = []
    to_drop = []
    for root in all_roots:
        columns = [c for c in df.columns if c.startswith(root)]
        mapping = dict(enumerate(c.replace(root + separator, "") for c in columns))
        exprs.append(
            pl.concat_list(columns).list.arg_max().map_dict(mapping).alias(root)
        )
        to_drop.extend(columns)

    return _dsds_with_columns_and_drop(df, exprs, to_drop)

def selective_one_hot_encode(
    df:PolarsFrame,
    selected: dict[str, list[str]],
    separator:str="_"
) -> PolarsFrame:
    '''
    Selectively one-hot encode only the provided values for the corresponding columns. This is equivalent to a 
    full one-hot encode followed by a column selection but this is more efficient. It is recommended to use this
    rather than full one-hot encoding after knowing the importance of each value in the string categories.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    selected
        A dictionary where the key refers to column names, and values are the values of the string column that 
        will be one-hot encoded. A value of None will be ignored.
    separator
        The separator used in the names of the new columns

    Example
    -------
    >>> import dsds.encoders as en
    ... df = pl.DataFrame({
    ...     "a": ["A", "B", "C", "D", "A"],
    ...     "b": ["AA", "BB", "BB", "CC", "CC"]
    ... })
    ... en.selective_one_hot_encode(df, {"a":["A", "B"], "b":["BB"]})
    shape: (5, 3)
    ┌─────┬─────┬──────┐
    │ a_A ┆ a_B ┆ b_BB │
    │ --- ┆ --- ┆ ---  │
    │ u8  ┆ u8  ┆ u8   │
    ╞═════╪═════╪══════╡
    │ 1   ┆ 0   ┆ 0    │
    │ 0   ┆ 1   ┆ 1    │
    │ 0   ┆ 0   ┆ 1    │
    │ 0   ┆ 0   ┆ 0    │
    │ 1   ┆ 0   ┆ 0    │
    └─────┴─────┴──────┘
    '''
    
    str_cols = list(selected.keys())
    _ = type_checker(df, str_cols, "string", "selective_one_hot_encode")

    exprs = []
    for c, vals in selected.items(): # Python's dict is ordered.
        exprs.extend(
            pl.col(c).eq(v).fill_null(False).cast(pl.UInt8).alias(c+separator+v) for v in vals if v is not None
        )
    return _dsds_with_columns_and_drop(df, exprs, str_cols)

def sum_encode(
    df: PolarsFrame
    , cols:Optional[list[str]]=None
    , separator:str="_"
) -> PolarsFrame:
    '''
    Sum encoding for the given string columns. Null will always be dropped. If a one-hot-encoded null indicator
    is desired, please see dsds.encoders.missing_indicator.
    
    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        If not provided, will use all string columns
    separator
        The separator used in the names of the new columns

    Reference
    ---------
    https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159
    https://datascience.stackexchange.com/questions/77941/what-is-sum-encoding
    '''
    if isinstance(cols, list):
        _ = type_checker(df, cols, "string", "sum_encode")
        str_cols = cols
    else:
        str_cols = get_string_cols(df)

    temp = df.lazy().select(
        pl.col(str_cols).unique().drop_nulls().implode().list.sort()
    )
    exprs:list[pl.Expr] = []
    for t in temp.collect().get_columns():
        u:pl.List = t[0] # t is a Series which contains a single series/list, so u is a series/list
        if len(u) > 1:
            exprs.extend(
                pl.when(pl.col(t.name).eq(u[0])).then(pl.lit(-1, dtype=pl.Int8)).otherwise(
                    pl.col(t.name).eq(u[i]).cast(pl.Int8)
                ).alias(t.name + separator + u[i])
                for i in range(1, len(u))
            )
        else:
            logger.info(f"During sum-encoding, the column {t.name} is found to have 1 unique value. Dropped.")
    
    return _dsds_with_columns_and_drop(df, exprs, str_cols)

def rank_hot_encode(
    df: PolarsFrame
    , col_ranks: dict[str, list[str]]
) -> PolarsFrame:
    '''
    Performs rank hot encoding. Currently this only supports string columns. Null values will remain 
    null in the encoded columns.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    col_ranks
        A dictionary with keys being name of columns, and values being a list of values in the column.
        The order of this list is used to represent the ranking of the values, and everything not inside
        the list will be considered to be 1s always. Only the categories given in the list will be 
        represented as columns. Note that >= lowest ranked value is always true. So that column is omitted.
        This also implies that if the list has length <=1, then the key value pair will be omitted.
    
    Example
    -------
    >>> import dsds.encoders as enc
    ... df = pl.DataFrame({
    ...     "test":["Very bad", "Bad", "Neutral", "Good", "Very good", "Good"],
    ...     "abc":["A", "B", "C", "A", "C", "C"]
    ... })
    ... # "Very bad" is the lowest rank. Everything is >= "Very bad", so it is omitted. Same for "A"
    >>> enc.rank_hot_encode(df, col_ranks={"test":["Very bad", "Bad", "Neutral", "Good"], "abc":["A", "B", "C"]})
    shape: (6, 5)
    ┌───────────┬───────────────┬────────────┬────────┬────────┐
    │ test>=Bad ┆ test>=Neutral ┆ test>=Good ┆ abc>=B ┆ abc>=C │
    │ ---       ┆ ---           ┆ ---        ┆ ---    ┆ ---    │
    │ u8        ┆ u8            ┆ u8         ┆ u8     ┆ u8     │
    ╞═══════════╪═══════════════╪════════════╪════════╪════════╡
    │ 0         ┆ 0             ┆ 0          ┆ 0      ┆ 0      │
    │ 1         ┆ 0             ┆ 0          ┆ 1      ┆ 0      │
    │ 1         ┆ 1             ┆ 0          ┆ 1      ┆ 1      │
    │ 1         ┆ 1             ┆ 1          ┆ 0      ┆ 0      │
    │ 1         ┆ 1             ┆ 1          ┆ 1      ┆ 1      │
    │ 1         ┆ 1             ┆ 1          ┆ 1      ┆ 1      │
    └───────────┴───────────────┴────────────┴────────┴────────┘
    '''
    ranks = {
        key:value
        for key, value in col_ranks.items() if len(value) > 1
    }
    cols = list(ranks.keys())
    _ = type_checker(df, cols, "string", "rank_hot_encode")

    ref_dicts = {
        k: {key:i for i, key in enumerate(r)}
        for k, r in ranks.items()
    }
    exprs = []
    for key, ref in ref_dicts.items():
        exprs.extend(
            pl.col(key).map_dict(ref, default=len(ref)).ge(i).cast(pl.UInt8).suffix(f">={v}")
            for v, i in ref.items() if v != ranks[key][0]
        )

    return _dsds_with_columns_and_drop(df, exprs, cols)

def force_binary(df:PolarsFrame) -> PolarsFrame:
    '''
    Force every binary column, no matter what data type, into 0s and 1s according to the order of the 
    elements. If a column has two unique values, like [null, "haha"], then null will be mapped 
    to 0 and "haha" to 1. This is not reversible.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    '''
    unique_cnt = get_unique_count(df).filter(pl.col("n_unique") == 2)
    binary_list = unique_cnt["column"].to_list()
    temp = df.lazy().select(
        pl.col(binary_list).unique().implode().list.sort()
    )
    exprs:list[pl.Expr] = [
        pl.col(t.name).eq(t[0][0]).cast(pl.UInt8).alias(t.name)
        for t in temp.collect().get_columns()
    ] # t is a Series which contains a single list which contains the 2 unique values 
    return _dsds_with_columns(df, exprs)

def multicat_one_hot_encode(
    df:PolarsFrame
    , cols: Optional[list[str]] = None
    , delimiter: str = "|"
    , drop_first: bool = True
) -> PolarsFrame:
    '''
    Expands multivalued categorical columns into several one-hot-encoded columns respectively. A multivalued categorical
    column is a column with strings like `aaa|bbb|ccc`, which means this row belongs to categories aaa, bbb, and ccc. 
    Typically, such a column will contain strings separated by a delimiter. This method will collect all unique strings 
    separated by the delimiter and one hot encode the corresponding column, e.g. by checking if `aaa` is contained in 
    values of this column. Null values will be mapped to 0 always. If a one-hot-encoded null indicator is desired, 
    please see dsds.encoders.missing_indicator.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        If not provided, will infer multicategorical columns.
    delimiter
        The delimiter in your multicategorical column
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
    if isinstance(cols, list):
        _ = type_checker(df, cols, "string", "multicat_one_hot_encode")
        multicats = cols
    else:
        multicats = infer_multicategorical(df, delimiter=delimiter)

    temp = df.lazy().select(
        pl.col(multicats).str.split(delimiter).explode().unique().drop_nulls().implode().list.sort()
    )
    exprs = []
    start_index = int(drop_first)
    for c in temp.collect().get_columns():
        u = c[0]
        if len(u) > 1:
            exprs.extend(
                pl.col(c.name).str.contains(u[i], literal=True).fill_null(False).cast(pl.UInt8).alias(
                    c.name + delimiter + u[i]
                )
                for i in range(start_index, len(u)) if isinstance(u[i], str)
            )
        else:
            logger.info(f"The multicategorical column {c.name} seems to have only 1 unique value. Dropped.")

    return _dsds_with_columns_and_drop(df, exprs, multicats)

def ordinal_auto_encode(
    df:PolarsFrame
    , cols:Optional[list[str]]=None
    , descending:bool = False
    , exclude:Optional[list[str]]=None
) -> PolarsFrame:
    '''
    Automatically applies ordinal encoding to the provided columns by the order of the elements. This method is 
    great for string columns like age ranges, with values like ["10-20", "20-30"], etc. (Beware of string lengths,
    e.g. if "100-110" exists in age range, then it may mess up the natural order.).

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

    temp = df.lazy().select(
        pl.col(ordinal_list).unique().implode().list.sort(descending=descending)
    )
    mappings = [ # map_dict on sorted uniques
        pl.col(t.name).map_dict(dict(zip(t[0], range(len(t[0])))))
        for t in temp.collect().get_columns()
    ]
    return _dsds_map_dict(df, mappings)
    
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
    mappings = []
    for c in ordinal_mapping:
        if c in df.columns:
            mapping = ordinal_mapping[c]
            mappings.append(pl.col(c).map_dict(mapping, default=default))
        else:
            logger.warning(f"Found that column {c} is not in df. Skipped.")

    return _dsds_map_dict(df, mappings)

def smooth_target_encode(
    df:PolarsFrame
    , target:str
    , cols:list[str]
    , min_samples_leaf:int = 20
    , smoothing:float = 10.
    , check_binary:bool=True
    , default: Optional[float] = None
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
    default
        If at transform time an unseen value orruces, it will be mapped to default
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

    # If c has null, null will become a group when we group by.
    lazy_references:list[pl.LazyFrame] = (
        df.lazy().group_by(c).agg(
            pl.count().alias("cnt"),
            pl.col(target).mean().alias("cond_p")
        ).with_columns(
            (1./(1. + ((-(pl.col("cnt").cast(pl.Float64) - min_samples_leaf))/smoothing).exp())).alias("alpha")
        ).select(
            pl.col(c),
            to = pl.col("alpha") * pl.col("cond_p") + (pl.lit(1) - pl.col("alpha")) * pl.lit(p)
        )
        for c in str_cols
    )
    mappings = [
        pl.col(str_cols[i]).map_dict(dict(zip(*ref.get_columns())), default=default)
        for i, ref in enumerate(pl.collect_all(lazy_frames=lazy_references))
    ]
    return _dsds_map_dict(df, mappings)

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
    return _dsds_with_columns(df, [pl.col(cols).cut(cuts).cast(pl.Utf8).suffix(suffix)])

# Can make it completely lazy
def fixed_sized_binning(
    df:PolarsFrame
    , cols:list[str]
    , interval: float
    , suffix:str = ""
) -> tuple[PolarsFrame, list[pl.Expr]]:
    '''
    Bins according to fixed interval size. The same cuts will be applied to all columns in cols. Bin will 
    start from min of feature to max of feature + interval with step length = interval.

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
    _ = type_checker(df, cols, "numeric", "fixed_sized_binning")
    bounds = df.lazy().select(
        pl.col(cols).min().prefix("min:")
        , pl.col(cols).max().prefix("max:")
    ).collect().row(0)
    exprs = []
    n = len(cols)
    for i, c in enumerate(cols):
        cut = np.arange(bounds[i], bounds[n+i] + interval, step=interval).tolist()
        exprs.append(pl.col(c).cut(cut).cast(pl.Utf8).suffix(suffix))

    return _dsds_with_columns(df, exprs)

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
    if isinstance(df, pl.LazyFrame):
        cuts = df.select(
            pl.col(cols).qcut(quantiles=n_bins).unique().cast(pl.Utf8).str.extract(r"\((.*?),")
            .cast(pl.Float64).sort().tail(n_bins)
        ).collect()
        exprs = [
            pl.col(c).cut(cuts.drop_in_place(c).to_list()).cast(pl.Utf8).suffix(suffix) for c in cols
        ]
    else: # Eager frame
        exprs = [pl.col(cols).qcut(quantiles=n_bins).cast(pl.Utf8).suffix(suffix)]

    return _dsds_with_columns(df, exprs)

def woe_cat_encode(
    df:PolarsFrame
    , target:str
    , cols:Optional[list[str]]=None
    , min_count:float = 1.
    , check_binary:bool = True
    , default: float = -10.
) -> PolarsFrame:
    '''
    Performs WOE encoding for categorical features. To WOE encode numerical columns, first bin them using
    custom_binning or quantile_binning. This only works for binary target. Nulls will be grouped as a category
    and mapped the WOE value for Nulls.

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
    check_binary : Might be moved to a global config in the future
        Whether to check target is binary or not.
    default
        Unseen values at transform time will be mapped to default
    '''
    if isinstance(cols, list):
        _ = type_checker(df, cols, "string", "woe_cat_encode")
        str_cols = cols
    else:
        str_cols = get_string_cols(df)

    if check_binary:
        if not check_binary_target(df, target):
            raise ValueError("Target is not binary or not properly encoded or contains nulls.")

    lazy_references = (
        df.lazy().group_by(s).agg(
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
        )
        for s in str_cols
    )
    mappings = [
        pl.col(str_cols[i]).map_dict(dict(zip(*ref.get_columns())), default=default)
        for i, ref in enumerate(pl.collect_all(lazy_frames=lazy_references))
    ]
    return _dsds_map_dict(df, mappings)