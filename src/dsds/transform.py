from __future__ import annotations

from .type_alias import (
    PolarsFrame
    , ImputationStrategy
    , ScalingStrategy
    , PowerTransformStrategy
    , clean_strategy_str
    , CPU_COUNT
)
from .prescreen import (
    get_bool_cols
    , get_string_cols
    , get_unique_count
    , check_binary_target
    , check_columns_types
)
from .blueprint import( # Need this for Polars extension to work
    Blueprint
)
import logging
import numpy as np
import polars as pl
from typing import Optional, Tuple, Any
from scipy.stats._morestats import (
    yeojohnson_normmax
    , boxcox_normmax
)
from concurrent.futures import as_completed, ThreadPoolExecutor
from tqdm import tqdm

# Rewrite all docstrings using Polars' format.

# A lot of companies are still using Python < 3.10
# So I am not using match statements

logger = logging.getLogger(__name__)

def impute(
    df:PolarsFrame
    , cols:list[str]
    , strategy:ImputationStrategy = 'median'
    , const:float = 1.
) -> PolarsFrame:
    '''Imputes the given columns using the given strategy.

        Arguments:
            df: either a eager/lazy Polars dataframe
            cols: cols to impute. If none, use all columns.
            strategy: one of 'mean', 'avg', 'average', 'median', 'const', 'constant', 'mode', 'most_frequent'. Some are 
            just alternative names for the same strategy.
            const: only uses this value if strategy = const

        Returns:
            the imputed lazy / eager dataframe.
    '''
    s = clean_strategy_str(strategy)

    # Given Strategy, define expressions
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

    # Need to cast to list so that pickle can work with it
    # This is unfortunate because we will be looping over this list twice... Whatever...
    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(list(exprs))
    return df.with_columns(exprs)

def scale(
    df:PolarsFrame
    , cols:list[str]
    , strategy:ScalingStrategy="normal"
    , const:float = 1.0
) -> PolarsFrame:
    '''
    Scale given columns using the given strategy. Will skip null values.

        Arguments:
            df: either a lazy or eager dataframe
            cols: list of columns to scale
            strategy: one of 'normal', 'standard', 'normalize', 'min_max', 'const', 'constant',
            where normal = standard = normalize.
            const: only uses this value if strategy = const

        Returns:
            the scaled lazy / eager dataframe.
    
    '''
    types = check_columns_types(df, cols)
    if types != "numeric":
        raise TypeError(f"Scaling can only be used on numeric columns, not {types} types.")

    s = clean_strategy_str(strategy)
    if s in ("normal", "standard", "normalize"):
        mean_std = df.select(cols).lazy().select(
            pl.all().mean().prefix("mean:")
            , pl.all().std().prefix("std:")
        ).collect().row(0)
        exprs = ( (pl.col(c) - mean_std[i])/(mean_std[i + len(cols)]) for i,c in enumerate(cols) )
    elif s == "min_max":
        min_max = df.select(cols).lazy().select(
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
    '''Converts all boolean columns into binary columns.

        Arguments:
            df: either a lazy or eager Polars DataFrame
            keep_null: if true, null will be kept. If false, null will be mapped to 0.

        Returns:
            a dataframe with booleans mapped to 0s and 1s.
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
    , cols: Optional[list[str]]
    , suffix: str = ":missing_ind"
) -> PolarsFrame:
    if cols is None:
        to_add = df.columns
    else:
        to_add = cols
    
    one = pl.lit(1, dtype=pl.UInt8)
    zero = pl.lit(0, dtype=pl.UInt8)
    exprs = (pl.when(pl.col(c).is_null()).then(one).otherwise(zero).alias(c+suffix) for c in to_add)

    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(list(exprs))
    return df.with_columns(exprs)

def one_hot_encode(
    df:PolarsFrame
    , cols:Optional[list[str]]=None
    , separator:str="_"
    , drop_first:bool=False
) -> PolarsFrame:
    '''One hot encoding. The separator must be a single character.'''
    
    if isinstance(cols, list):
        types = check_columns_types(df, cols)
        if types != "string":
            raise TypeError(f"One-hot encoding can only be used on string columns, not {types} types.")
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
    , exclude:Optional[list[str]]=None
) -> PolarsFrame:
    '''
    Encode the given columns as binary values. Only handles string binary at this moment. This is equivalent
    to using one-hot-encoding on binary columns using drop_first=True.

        Arguments:
            df:
            binary_cols: the binary_cols you wish to convert. If no input, will infer.
            exclude: the columns you wish to exclude in this transformation. 

        Returns: 
            (the transformed dataframe, mapping table between old values to [0,1])
    '''

    if cols is None:
        str_cols = get_string_cols(df)
        exclude = [] if exclude is None else exclude
        binary_list = get_unique_count(df)\
            .filter( # Binary + Not Exclude + Only String
                (pl.col("n_unique") == 2) & (~pl.col("column").is_in(exclude)) & (pl.col("column").is_in(str_cols))
            ).get_column("column").to_list()

    else: # No need to do all that type checking steps because we are gonna do that in one-hot anyways
        binary_list = cols
    
    return one_hot_encode(df, cols=binary_list, drop_first=True)

def multicat_one_hot_encode(
    df:PolarsFrame
    , cols: list[str]
    , delimiter: str
    , drop_first: bool = False
):
    '''
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

def force_binary(
    df:PolarsFrame
    , cols:Optional[list[str]]=None
):
    '''
    Force every binary column, no matter what data type, to be turned into 0s and 1s by the order of the elements. If a 
    column has two unique values like [null, "haha"], then null will be mapped to 0 and "haha" to 1.
    '''
    if cols is None:
        binary_list = get_unique_count(df)\
            .filter(pl.col("n_unique") == 2)\
            .get_column("column").to_list()
    else:
        binary_list = cols

    temp = df.lazy().select(binary_list).groupby(1).agg(
            pl.all().unique().sort()
        ).select(binary_list) # Need this to get rid of the literal 1 column
    exprs:list[pl.Expr] = []
    one = pl.lit(1, dtype=pl.UInt8) # Avoid casting 
    zero = pl.lit(0, dtype=pl.UInt8) # Avoid casting
    for t in temp.collect().get_columns():
        u:pl.List = t[0] # t is a Series which contains a single list which contains the 2 unique values 
        if len(u) == 2:
            exprs.append(
                pl.when(pl.col(t.name) == u[0]).then(zero).otherwise(one).alias(t.name)
            )
        else:
            logger.info(f"During force_binary, the column {t.name} is found to have != 2 unique values. Ignored.")
    
    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(exprs)
    return df.with_columns(exprs)    

def get_mapping_table(ordinal_mapping:dict[str, dict[str,int]]) -> pl.DataFrame:
    '''
        Helper function to get a table from an ordinal_mapping dict.

        >>> {
        >>> "a": 
        >>>    {"a1": 1, "a2": 2,},
        >>> "b":
        >>>    {"b1": 3, "b2": 4,},
        >>> }


        Arguments:
            ordinal_mapping: {name_of_feature: {value_1 : mapped_to_number_1, value_2 : mapped_to_number_2, ...}, ...}

        Returns:
            A table with feature name, value, and mapped_to
    
    '''
    mapping_tables:list[pl.DataFrame] = []
    for feature, mapping in ordinal_mapping.items():
        table = pl.from_records(list(mapping.items()), schema=["value", "mapped_to"]).with_columns(
            pl.lit(feature).alias("feature")
        ).select("feature", "value", "mapped_to")
        mapping_tables.append(table)

    return pl.concat(mapping_tables)

def ordinal_auto_encode(
    df:PolarsFrame
    , cols:Optional[list[str]]=None
    , descending:bool = False
    , exclude:Optional[list[str]]=None
) -> PolarsFrame:
    '''
        Automatically applies ordinal encoding to the provided columns by the following logic:
            Sort the column, smallest value will be assigned to 0, second smallest will be assigned to 1...

        This will automatically detect string columns and apply this operation if ordinal_cols is not provided. 
        This method is great for string columns like age ranges, with values like ["10-20", "20-30"], etc...
        
        Arguments:
            df:
            default:
            cols:
            exclude: the columns you wish to exclude in this transformation.
        
        Returns:
            (encoded df, mapping table)
    '''
    if isinstance(cols, list):
        types = check_columns_types(df, cols)
        if types != "string":
            raise TypeError(f"Ordinal encoding can only be used on string columns, not {types} types.")
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
        Ordinal encode the data with given mapping.

        Notice that this function assumes that you already have the mapping, in correct mapping format.
        since you have to supply the ordinal_mapping argument. If you still want the tabular output format,
        please call get_ordinal_mapping_table with ordinal_mapping, which will create a table from this.

        Arguments:
            df:
            ordinal_mapping:
            default: if a value for a feature does not exist in ordinal_mapping, use default.

        Returns:
            encoded df
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
    '''Smooth target encoding for binary classification. Currently only implemented for binary target.

        See https://towardsdatascience.com/dealing-with-categorical-variables-by-using-target-encoder-a0f1733a4c69

        Arguments:
            df:
            target:
            cat_cols:
            min_samples_leaf:
            smoothing:
            check_binary:
    '''
    if isinstance(cols, list):
        types = check_columns_types(df, cols)
        if types != "string":
            raise ValueError(f"Target encoding can only be used on string columns, not {types} types.")
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

def _when_then_replace(c:str, repl_map:dict):
    expr = pl.col(c)
    for code, repl in repl_map.items():
        expr = pl.when(pl.col(c).eq(code)).then(repl).otherwise(expr)
    
    return expr.alias(c)

def feature_mapping(
    df:PolarsFrame
    , mapping: dict[str, dict[Any, Any]] | list[pl.Expr] | pl.Expr
) -> PolarsFrame:
    '''
    Maps specific values of a feature into values provided. This is a common task when the feature columns come with 
    error codes.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    mapping
        Either a dict like {"a": {999: None, 998: None, 997: None}, ...}, meaning that 999, 998 and 997 in column "a" 
        should be replaced by null, or a list/a single Polars (when-then) expression(s) like the following,  
        pl.when(pl.col("a") >= 997).then(None).otherwise(pl.col("a")).alias("a"), which will perform the same mapping 
        as the dict example. Note that using Polars expression can tackle more complex replacement.

    Returns
    -------
        Either a lazy or eager Polars dataframe with specific values of features replaced

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
            exprs.append(_when_then_replace(c, repl_map))
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
) -> PolarsFrame:
    '''
    Bins according to the cuts provided. The same cuts will be applied to all columns in cols.
    '''

    if isinstance(df, pl.LazyFrame):
        exprs = [
            pl.col(c).cut(cuts).cast(pl.Utf8) for c in cols
        ]
        return df.blueprint.with_columns(exprs)
    else:
        return df.with_columns(
            pl.col(c).cut(cuts).cast(pl.Utf8) for c in cols
        )

def quantile_binning(
    df:PolarsFrame
    , cols:list[str]
    , n_bins:int
) -> PolarsFrame:
    '''
    Bin a continuous variable into categories, based on quantile. Null values will be its own category. The same binning
    rule will be applied to all columns in cols. If you want different n_bins for different columns, chain another 
    quantile_binning with different cols and n_bins.

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
    
    Returns
    -------
        A lazy or eager frame with the columns binned.

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
    types = check_columns_types(df, cols)
    if types != "numeric":
        raise ValueError(f"Quantile binning can only be used on numeric columns, not {types} types.")
    
    qcuts = np.arange(start=1/n_bins, stop=1.0, step = 1/n_bins)
    if isinstance(df, pl.LazyFrame):
        cuts = df.select(cols).select(
            pl.all().qcut(qcuts).unique().cast(pl.Utf8).str.extract(r"\((.*?),")
            .cast(pl.Float64).sort().tail(len(qcuts))
        ).collect()
        # For some reasons a generator here doesn't work. Use list instead
        exprs = [
            pl.col(c).cut(cuts.drop_in_place(c).to_list()).cast(pl.Utf8) for c in cols
        ]
        return df.blueprint.with_columns(exprs)
    else: # Eager frame
        return df.with_columns(
            pl.col(c).qcut(qcuts).cast(pl.Utf8) for c in cols 
        )

def woe_cat_encode(
    df:PolarsFrame
    , target:str
    , cols:Optional[list[str]]=None
    , min_count:float = 1.
    , default: float = -10.
    , check_binary:bool = True
) -> PolarsFrame:
    '''Performs WOE encoding for categorical features. Currently woe encoding is only available
    for categorical (string) features. Numerical WOE encoding requires binning and the binning
    transform is being considered (Need it to be comptible with blueprints and all that).

        Arguments:
            df: either a lazy or eager dataframe
            target: target column
            cols: string columns to be encoded. If none, it will use all from the df.
            min_count: a regularization term that prevents ln(0).
            default: default value for nulls resulted in the left-join.
            check_binary: whether to check if target is binary or not

        Returns:
            an woe encoded lazy or eager Polars dataframe
    '''
    if isinstance(cols, list):
        types = check_columns_types(df, cols)
        if types != "string":
            raise ValueError(f"woe_cat_encode encoding can only be used on string columns, not {types} types.")
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
    '''Performs power transform on the numerical columns.

        Arguments:
            df: either a lazy or eager Polars dataframe
            cols: a list of numerical columns to perform the transform.
            strategy: either yeo_johnson or box_cox
            n_threads: max number of threads you want to use. Default = CPU_COUNT

        Returns:
            the transformed lazy or eager dataframe
    '''

    types = check_columns_types(df, cols)
    if types != "numeric":
        raise ValueError(f"Power Transform can only be used on numeric columns, not {types} types.")
    
    s = clean_strategy_str(strategy)
    exprs:list[pl.Expr] = []
    # Ensure columns do not have missing values
    exclude_columns_w_nulls = df.lazy().select(cols).null_count().collect().transpose(
        include_header=True, column_names=["null_count"]
    ).filter(pl.col("null_count") > 0).get_column("column").to_list()

    if len(exclude_columns_w_nulls) > 0:
        logger.info("The following columns will not be processed by power_transform because they contain missing "
                    f"values. Please impute them.\n{exclude_columns_w_nulls}")
        
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

    