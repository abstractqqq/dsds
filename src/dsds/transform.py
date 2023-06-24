from __future__ import annotations

from .type_alias import (
    PolarsFrame
    , ImputationStrategy
    , ScalingStrategy
    , clean_strategy_str
)
from .prescreen import (
    get_bool_cols
    , get_numeric_cols
    , get_string_cols
    , get_unique_count
    , dtype_mapping
)

import logging
import polars as pl
from typing import Optional

# A lot of companies are still using Python < 3.10
# So I am not using match statements
# Well, it does say in project description that we need Python 3.10.

logger = logging.getLogger(__name__)


def check_columns_types(df:PolarsFrame, cols:Optional[list[str]]=None) -> str:
    '''Returns the unique types of given columns in a single string. If multiple types are present
    they are joined by a |. If cols is not given, automatically uses all df's columns.'''
    types = set()
    if cols is None:
        check_cols:list[str] = df.columns
    else:
        check_cols:list[str] = cols 

    temp = df.select(check_cols)
    for t in temp.dtypes:
        types.add(dtype_mapping(t))
    
    return "|".join(types) if len(types) > 0 else "unknown"

# def create_map_expr(
#         col_name:str
#         , gen:Generator[Tuple[str, Any], None, None]
#         , default:Any = None
# ) -> pl.Expr:

#     '''
#         Suppose you have a dictionary like d = {"a":1, "b":2}. Instead of doing pl.col("column").map_dict(d), you can 
#         do pl.when(pl.col("column") == "a").then(1).otherwise(pl.when(pl.col("column") == "b").then(2).otherwise(default))
#         instead. This function generators this expression for you from a generator that yields a key value pair.
    
#     '''

#     next_pair = next(gen, None)
#     if next_pair:
#         k, v = next_pair
#         return pl.when(pl.col(col_name) == k).then(v).otherwise(create_map_expr(col_name, gen))
#     return pl.lit(default)

def impute(df:PolarsFrame
    , cols:list[str]
    , strategy:ImputationStrategy = 'median'
    , const:float = 1.
) -> PolarsFrame:
    '''
        Arguments:
            df:
            cols:
            strategy:
            const: only uses this value if strategy = ImputationStartegy.CONST
    
    '''
    s = clean_strategy_str(strategy)
    # Given Strategy, define expressions
    if s == "median":
        all_medians = df.lazy().select(cols).median().collect().to_numpy().ravel()
        exprs = (pl.col(c).fill_null(all_medians[i]) for i,c in enumerate(cols))
    elif s in ("mean", "avg", "average"):
        all_means = df.lazy().select(cols).mean().collect().to_numpy().ravel()
        exprs = (pl.col(c).fill_null(all_means[i]) for i,c in enumerate(cols))
    elif s in ("const", "constant"):
        exprs = (pl.col(c).fill_null(const) for c in cols)
    elif s in ("mode", "most_frequent"):
        all_modes = df.lazy().select(pl.col(c).mode() for c in cols).collect().to_numpy().ravel()
        exprs = (pl.col(c).fill_null(all_modes[i]) for i,c in enumerate(cols))
    else:
        raise TypeError(f"Unknown imputation strategy: {strategy}")

    return df.with_columns(exprs)

def scale(df:PolarsFrame
    , cols:list[str]
    , strategy:ScalingStrategy="normal"
    , const:float = 1.0
) -> PolarsFrame:
    
    '''
        Arguments:
            df:
            cols:
            strategy:
            const: only uses this value if strategy = ImputationStartegy.CONST
    
    '''
    types = check_columns_types(df, cols)
    if types != "numeric":
        raise ValueError(f"Scaling can only be used on numeric columns, not {types} types.")

    s = clean_strategy_str(strategy)
    if s in ("normal", "standard", "normalize"):
        all_means = df.lazy().select(cols).mean().collect().to_numpy().ravel()
        all_stds = df.lazy().select(cols).std().collect().to_numpy().ravel()
        exprs = ( (pl.col(c) - all_means[i])/(all_stds[i]) for i,c in enumerate(cols) )
    elif s == "min_max":
        all_mins = df.lazy().select(cols).min().collect().to_numpy().ravel()
        all_maxs = df.lazy().select(cols).max().collect().to_numpy().ravel()
        exprs = ( (pl.col(c) - all_mins[i])/((all_maxs[i] - all_mins[i])) for i,c in enumerate(cols) )
    elif s in ("const", "constant"):
        exprs = (pl.col(c)/const for c in cols)
    else:
        raise TypeError(f"Unknown scaling strategy: {strategy}")

    # transformed = df.with_columns(exprs)
    return df.with_columns(exprs)

def boolean_transform(df:PolarsFrame, keep_null:bool=True) -> PolarsFrame:
    '''
        Converts all boolean columns into binary columns.
        Arguments:
            df:
            keep_null: if true, null will be kept. If false, null will be mapped to 0.

    '''
    bool_cols = get_bool_cols(df)
    if keep_null: # Directly cast. If null, then cast will also return null
        exprs = (pl.col(c).cast(pl.UInt8) for c in bool_cols)
    else: # Cast. Then fill null to 0s.
        exprs = (pl.col(c).cast(pl.UInt8).fill_null(0) for c in bool_cols)

    return df.with_columns(exprs)

def one_hot_encode(
    df:PolarsFrame
    , cols:Optional[list[str]]=None
    , separator:str="_"
    , drop_one:bool=False
) -> PolarsFrame:
    '''One hot encoding. The separator must be a single character.'''
    
    str_cols = []
    if isinstance(cols, list):
        types = check_columns_types(df, cols)
        if types != "string":
            raise ValueError(f"One-hot encoding can only be used on string columns, not {types} types.")
        str_cols.extend(cols)
    else:
        str_cols = get_string_cols(df)

    temp = df.lazy().groupby(1).agg(
        pl.col(s).unique().sort() for s in str_cols
    ).select(str_cols)
    exprs:list[pl.Expr] = []
    for t in temp.collect().get_columns():
        uniques:pl.Series = t[0] # t is a Series which contains one subseries
        if drop_one:
            uniques = uniques.slice(offset=1)
        u: str
        for u in uniques: # u is a string
            exprs.append( pl.when(pl.col(t.name) == u).then(1).otherwise(0).alias(t.name + separator + u) )

    return df.with_columns(exprs).drop(str_cols)

# def fixed_sized_encode(df:pl.DataFrame, num_cols:list[str], bin_size:int=50) -> TransformationResult:
#     '''Given a continuous variable, take the smallest `bin_size` of them, and call them bin 1, take the next
#     smallest `bin_size` of them and call them bin 2, etc...
    
#     '''
#     pass

def percentile_encode2():
    pass 

# REWRITE THIS
# def percentile_encode(df:pl.DataFrame
#     , cols:list[str]=None
#     , exclude:list[str]=None
# ) -> FitTransform:
#     '''Bin your continuous variable X into X_percentiles. This will create at most 100 + 1 bins, 
#         where each percentile could potentially be a bin and null will be mapped to bin = 0. 
#         Bin 1 means percentile 0 to 1. Generally, bin X groups the population from bin X-1 to 
#         bin X into one bucket.

#         I see some potential optimization opportunities here.

#         Arguments:
#             df:
#             num_cols: 
#             exclude:

#         Returns:
#             (A transformed dataframe, a mapping table (value to percentile))
    
#     '''

#     # Percentile Binning

#     num_list:list[str] = []
#     exclude_list:list[str] = [] if exclude is None else exclude
#     if isinstance(cols, list):
#         types = check_columns_types(df, cols)
#         if types != "numeric":
#             raise ValueError(f"Percentile encoding can only be used on numeric columns, not {types} types.")
#         num_list.extend(cols)
#     else:
#         num_list.extend(get_numeric_cols(df, exclude=exclude_list))

#     exprs:list[pl.Expr] = []
#     all_mappings = []
#     for c in num_list:
#         percentile = df.groupby(c).agg(pl.count().alias("cnt"))\
#             .sort(c)\
#             .with_columns(
#                 ((pl.col("cnt").cumsum()*100)/len(df)).ceil().alias("percentile")
#             ).groupby("percentile")\
#             .agg(
#                 pl.col(c).min().alias("min"),
#                 pl.col(c).max().alias("max"),
#                 pl.col("cnt").sum().alias("cnt"),
#             ).sort("percentile").select(
#                 pl.lit(c).alias("feature"),
#                 pl.col("percentile").cast(pl.UInt8),
#                 "min",
#                 "max",
#                 "cnt",
#             )
        
#         first_row = percentile.select("percentile","min", "max").to_numpy()[0, :] # First row
#         # Need to handle an extreme case when percentile looks like 
#         # percentile   min   max
#         #  p1         null  null
#         #  p2          ...   ...
#         # This happens when there are so many nulls in the column.
#         if np.isnan(first_row[2]):
#             # Discard the first row if this is the case. 
#             percentile = percentile.slice(1, length = None)

#         # Only work on non null values. Null will be mapped to default value anyway.
#         temp_df = df.lazy().filter(pl.col(c).is_not_null()).sort(c).set_sorted(c)\
#             .join_asof(other=percentile.lazy().set_sorted("max"), left_on=c, right_on="max", strategy="forward")\
#             .select(c, "percentile")\
#             .unique().collect()
        
#         real_mapping = dict(zip(temp_df[c], temp_df["percentile"]))
#         # a representation of the mapping, needed for recreating this.
#         repr_mapping = dict(zip(percentile["max"], percentile["percentile"]))
#         all_mappings.append(repr_mapping)
#         exprs.append(
#             pl.col(c).map_dict(real_mapping, default=0).cast(pl.UInt8)
#         )

#     res = df.with_columns(exprs)
#     encoder_rec = EncoderRecord(features=num_list, strategy=EncodingStrategy.PERCENTILE, mappings=all_mappings)
#     return FitTransform(transformed=res, mapping=encoder_rec)

def binary_encode(df:PolarsFrame
    , cols:Optional[list[str]]=None
    , exclude:Optional[list[str]]=None
) -> PolarsFrame:
    
    '''Encode the given columns as binary values. Only hands string binaries at this moment.

        The goal of this function is to map binary string values into [0, 1], therefore reducing the amount of encoding
        you will have to do later. The values will be mapped to [0, 1] by the following rule:
            if value_1 < value_2, value_1 --> 0, value_2 --> 1. E.g. 'N' < 'Y' ==> 'N' --> 0 and 'Y' --> 1
        
        In case the two distinct values are [None, value_1], and you decide to treat this variable as a binary category
        , then None --> 0 and value_1 --> 1. 
        
        Using one-hot-encoding will map binary categorical values to 2 columns (except when you specify drop_first=True 
        in pd.get_dummies), therefore introducing unnecessary dimension. So it is better to prevent it.

        If case the distinct values are [null, value_1, value_2], then this is not currently considered as a 
        binary column.

        Arguments:
            df:
            binary_cols: the binary_cols you wish to convert. If no input, will infer.
            exclude: the columns you wish to exclude in this transformation. 

        Returns: 
            (the transformed dataframe, mapping table between old values to [0,1])
    '''

    binary_list = []
    if isinstance(cols, list):
        binary_list.extend(cols)
    else:
        str_cols = get_string_cols(df)
        exclude = [] if exclude is None else exclude
        binary_columns = get_unique_count(df)\
            .filter( # Binary + Not Exclude + Only String
                (pl.col("n_unique") == 2) & (~pl.col("column").is_in(exclude)) & (pl.col("column").is_in(str_cols))
            ).get_column("column")

        # Binary numericals are kept the way they are.
        binary_list.extend(binary_columns)     
    
    temp = df.lazy().groupby(1).agg(
        pl.col(b).unique().sort() for b in binary_list
    ).select(binary_list) # Null will be first in the sort.
    exprs:list[pl.Expr] = []
    for t in temp.collect().get_columns():
        s:pl.Series = t[0] # t is a len 1 series that contains another series. So s = t[0] will get the series out. 
        # s has 2 elements.
        if len(s) == 2: # s is a pl.Series, s is already sorted, and null will come first
            exprs.append(pl.when(pl.col(t.name) == s[0]).then(0).otherwise(1).cast(pl.UInt8).alias(t.name))
        else:
            logger.warning(f"Found {s.name} column has {len(s)} unique values instead of 2. Ignored.")

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
    , cols:list[str]=None
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
            ordinal_cols:
            exclude: the columns you wish to exclude in this transformation. (Only applies if you are letting the system auto-detecting columns.)
        
        Returns:
            (encoded df, mapping table)
    '''
    ordinal_list:list[str] = []
    if isinstance(cols, list):
        types = check_columns_types(df, cols)
        if types != "string":
            raise ValueError(f"Ordinal encoding can only be used on string columns, not {types} types.")
        ordinal_list.extend(cols)
    else:
        ordinal_list.extend(get_string_cols(df, exclude=exclude))

    is_input_lazy = isinstance(df, pl.LazyFrame)
    temp = df.lazy().groupby(1).agg(
        pl.col(c).unique().sort() for c in ordinal_list
    ).select(ordinal_list)
    duplicates = []
    for t in temp.collect().get_columns():
        uniques:pl.Series = t[0]
        ordinal_col_name = t.name + "_ordinal"
        duplicates.append(ordinal_col_name)
        temp_table = pl.LazyFrame((uniques, pl.Series(range(len(uniques)))), schema=[t.name, ordinal_col_name])
        if not is_input_lazy:
            temp_table = temp_table.collect()
        # Use join instead of map_dict because LazyFrame's write_json does not work with map_dict.
        df = df.join(temp_table, on = t.name, how="left").with_columns(
                pl.col(ordinal_col_name).alias(t.name)
            )

    return df.drop(duplicates)

def ordinal_encode(df:PolarsFrame
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
    
    is_lazy_input = isinstance(df, pl.LazyFrame)
    duplicates = []
    for c in ordinal_mapping:
        if c in df.columns:
            mapping = ordinal_mapping[c]
            new_col_name = c + "_to"
            duplicates.append(new_col_name)
            temp_df = pl.LazyFrame((mapping.keys(), mapping.values()), schema=[c, new_col_name])
            if not is_lazy_input:
                temp_df = temp_df.collect()

            df = df.join(temp_df, on = c, how="left").with_columns(
                pl.col(new_col_name).fill_null(default).alias(c)
            )
        else:
            logger.warning(f"Found that column {c} is not in df. Skipped.")

    return df.drop(duplicates)

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
    str_cols:list[str] = []
    if isinstance(cols, list):
        types = check_columns_types(df, cols)
        if types != "string":
            raise ValueError(f"Target encoding can only be used on string columns, not {types} types.")
        str_cols.extend(cols)
    else:
        str_cols = get_string_cols(df)
    
    # Only works for binary target for now 
    # Check if it is binary or not.
    if check_binary:
        target_uniques = df.lazy().select(pl.col(target).unique()).collect().get_column(target)
        if len(target_uniques) != 2 or (not (0 in target_uniques and 1 in target_uniques)):
            raise ValueError(f"The target column {target} must be a binary target with 0s and 1s.")

    p = df.lazy().select(pl.col(target).mean()).collect().to_numpy()[0,0] # probability of target = 1
    # If c has null, null will become a group when we group by.
    duplicates = []
    for c in str_cols:
        new_name = c + "_encoded_as"
        duplicates.append(new_name)
        ref = df.groupby(c).agg(
            pl.count().alias("cnt"),
            pl.col(target).mean().alias("cond_p")
        ).with_columns(
            (1./(1. + ((-(pl.col("cnt").cast(pl.Float64) - min_samples_leaf))/smoothing).exp())).alias("alpha")
        ).select(
            pl.col(c),
            (pl.col("alpha") * pl.col("cond_p") + (pl.lit(1) - pl.col("alpha")) * pl.lit(p)).alias(new_name)
        ) # If df is lazy, ref is lazy. If df is eager, ref is eager

        # It is ok to do inner join because all values of c are present in ref.
        df = df.join(ref, on = c).with_columns(
            pl.col(new_name).alias(c)
        )
        
    return df.drop(duplicates)

def power_transform():
    pass