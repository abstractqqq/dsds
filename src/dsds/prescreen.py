
from .type_alias import (
    PolarsFrame
)

import polars.selectors as cs
import polars as pl 
import re
import logging  
from datetime import datetime 
from typing import Final, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)
POLARS_NUMERICAL_TYPES:Final[list[pl.DataType]] = [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]  # noqa: E501
POLARS_DATETIME_TYPES:Final[list[pl.DataType]] = [pl.Datetime, pl.Date, pl.Time]

#----------------------------------------------------------------------------------------------#
# Generic columns checks | Only works with Polars because Pandas's data types suck!            #
#----------------------------------------------------------------------------------------------#

def get_numeric_cols(df:PolarsFrame, exclude:list[str]=None) -> list[str]:
    output = []
    exclude_list = [] if exclude is None else exclude
    for c,t in zip(df.columns, df.dtypes):
        if t in POLARS_NUMERICAL_TYPES and c not in exclude_list:
            output.append(c)
    return output

def get_string_cols(df:PolarsFrame, exclude:list[str]=None) -> list[str]:
    output = []
    exclude_list = [] if exclude is None else exclude
    for c,t in zip(df.columns, df.dtypes):
        if t == pl.Utf8 and c not in exclude_list:
            output.append(c)
    return output

def get_datetime_cols(df:PolarsFrame) -> list[str]:
    '''Only gets datetime columns, will not infer from strings.'''
    return [c for c,t in zip(df.columns, df.dtypes) if t in POLARS_DATETIME_TYPES]

def get_bool_cols(df:PolarsFrame) -> list[str]:
    return [c for c,t in zip(df.columns, df.dtypes) if t == pl.Boolean]

def get_cols_regex(df:PolarsFrame, pattern:str, lowercase:bool=False) -> list[str]:
    reg = re.compile(pattern)
    if lowercase:
        return [f for f in df.columns if reg.search(f)]
    return [f for f in df.columns if reg.search(f.lower())]

# dtype can be a "pl.datatype" or just some random data for which we want to infer a generic type.
def dtype_mapping(d: Any) -> str:
    if isinstance(d, str) or d == pl.Utf8:
        return "string"
    elif isinstance(d, bool) or d == pl.Boolean:
        return "bool"
    elif isinstance(d, (int,float)) or d in POLARS_NUMERICAL_TYPES:
        return "numeric"
    elif isinstance(d, datetime) or d in POLARS_DATETIME_TYPES:
        return "datetime"
    else:
        return "other/unknown"
    
#----------------------------------------------------------------------------------------------#
# Prescreen Inferral, Removal Methods                                                          #
#----------------------------------------------------------------------------------------------#

@dataclass
class DroppedFeatureResult:
    dropped: list[str]
    reason: str

    # todo!
    def __str__(self):
        pass 

# Add a slim option that returns fewer stats. This is generic describe.
def describe(df:pl.DataFrame) -> pl.DataFrame:
    '''
        The transpose view of df.describe() for easier filtering. Add more statistics in the future (if easily
        computable.)

        Arguments:
            df:

        Returns:
            Transposed view of df.describe() with a few more interesting columns
    '''
    
    temp = df.describe()
    desc = temp.drop_in_place("describe")
    unique_counts = get_unique_count(df).with_columns(
        (pl.col("n_unique") / len(df)).alias("unique_pct"),
        pl.when(pl.col("n_unique")==2).then(1).otherwise(0).alias("is_binary")
    )

    skew_and_kt = df.select(pl.col(c).skew() for c in df.columns)\
                .transpose(include_header=True, column_names=["skew"])\
                .join(
                    df.select(pl.col(c).kurtosis() for c in df.columns)\
                    .transpose(include_header=True, column_names=["kurtosis"])
                , on = "column")

    nums = ("count", "null_count", "mean", "std", "median", "25%", "75%")
    dtypes_dict = dict(zip(df.columns, map(dtype_mapping, df.dtypes)))
    final = temp.transpose(include_header=True, column_names=desc).with_columns(
        (pl.col(c).cast(pl.Float64) for c in nums)
    ).with_columns(
        (pl.col("null_count")/pl.col("count")).alias("null_pct"),
        pl.col("column").map_dict(dtypes_dict).alias("dtype")
    ).join(unique_counts, on="column").join(skew_and_kt, on="column")
    
    return final.select(('column', 'is_binary','count','null_count','null_pct','n_unique'
                        , 'unique_pct','mean','std','min','max','25%'
                        , 'median','75%', "skew", "kurtosis",'dtype'))

# Numeric only describe. Be more detailed.

# String only describe. Be more detailed about interesting string stats.

def describe_str(df:PolarsFrame, words_to_count:Optional[list[str]]=None) -> pl.DataFrame:
    '''Gives some statistics about the string columns. Optionally you may pass a list
    of strings to compute the total occurrances of each of the words in the string columns.

    '''
    strs = get_string_cols(df)
    df_str = df.select(strs)
    if isinstance(df, pl.LazyFrame):
        df_str = df_str.collect()

    nc = df_str.null_count()\
        .transpose().to_series().rename("null_count")
    smax = df_str.select(
        pl.col(c).max() for c in strs
    ).transpose().to_series().rename("max")
    smin = df_str.select(
        pl.col(c).min() for c in strs
    ).transpose().to_series().rename("min")
    mode = df_str.select(
        pl.col(c).mode().take(0) for c in strs
    ).transpose().to_series().rename("mode")
    mins = df_str.select(
        pl.col(c).str.lengths().min() for c in strs
    ).transpose().to_series().rename("min_byte_len")
    maxs = df_str.select(
        pl.col(c).str.lengths().max() for c in strs
    ).transpose().to_series().rename("max_byte_len")
    avgs = df_str.select(
        pl.col(c).str.lengths().mean() for c in strs
    ).transpose().to_series().rename("avg_byte_len")
    medians = df_str.select(
        pl.col(c).str.lengths().median() for c in strs
    ).transpose().to_series().rename("median_byte_len")
    space_counts = df_str.select(
        pl.col(c).str.count_match(r"\s").mean() for c in strs
    ).transpose().to_series().rename("avg_space_cnt")
    nums_counts = df_str.select(
        pl.col(c).str.count_match(r"[0-9]").mean() for c in strs
    ).transpose().to_series().rename("avg_digit_cnt")
    cap_counts = df_str.select(
        pl.col(c).str.count_match(r"[A-Z]").mean() for c in strs
    ).transpose().to_series().rename("avg_cap_cnt")
    lower_counts = df_str.select(
        pl.col(c).str.count_match(r"[a-z]").mean() for c in strs
    ).transpose().to_series().rename("avg_lower_cnt")

    output = {
        "features":strs,
        nc.name: nc,
        "null_pct": nc/len(df),
        smin.name: smin,
        smax.name: smax,
        mode.name: mode,
        mins.name: mins,
        maxs.name: maxs,
        avgs.name: avgs,
        medians.name: medians,
        space_counts.name: space_counts,
        nums_counts.name: nums_counts,
        cap_counts.name: cap_counts,
        lower_counts.name: lower_counts
    }

    if isinstance(words_to_count, list):
        for w in words_to_count:
            if isinstance(w, str):
                t = df_str.select(
                    pl.col(c).str.count_match(w).sum() for c in strs
                ).transpose().to_series().rename("total_"+ w + "_count")
                output[t.name] = t

    return pl.from_dict(output) 

def non_numeric_removal(df:PolarsFrame, include_bools:bool=True) -> PolarsFrame:
    '''Removes all non-numeric columns. If include_bools = True, then keep boolean columns.'''
    
    nums = get_numeric_cols(df)
    if include_bools:
        nums += get_bool_cols(df)
    non_nums = [c for c in df.columns if c not in nums]
    logger.info(f"The following columns are dropped because they are not numeric: {non_nums}.\n"
                f"Removed a total of {len(non_nums)} columns.")
    return df.drop(non_nums)

# Check if column follows the normal distribution. Hmm...
def normal_inferral():
    pass

# Check if columns are duplicates. Might take time.
def duplicate_inferral():
    # Get profiles first.
    # Divide into categories: bools, strings, numerics, datetimes.
    # Then cut down list to columns that have the same min, max, n_unique and null_count.
    # Then check equality..
    pass

def pattern_inferral():
    pass

# Check if column is an email column. This is easy. Email regex is easy.
# But email column might have many nulls. Check only on non-null values.
def email_inferral():
    pass 

# Check for columns that are US zip codes.
def zipcode_inferral():
    pass

# Check if a column is date
def date_inferral(df:PolarsFrame) -> list[str]:
    '''Infers date columns in dataframe. This inferral is not perfect.'''
    logger.info("Date Inferral is error prone due to the huge variety of date formats. Please use with caution.")
    
    dates = [c for c,t in zip(df.columns, df.dtypes) if t in POLARS_DATETIME_TYPES]
    strings = get_string_cols(df)

    # Might be memory intensive on big dataframes. 
    sample_df = df.lazy().select(strings)\
        .drop_nulls().collect()\
        .sample(n = 1000).select(
            # Cleaning the string first. Only try to catch string dates which are in the first split by space
           pl.col(s).str.strip().str.replace_all("(/|\.)", "-").str.split(by=" ").list.first() 
           for s in strings
        )
    for s in strings:
        try:
            c = sample_df[s].str.to_date(strict=False)
            if 1 - c.null_count()/1000 >= 0.15: # if at least 15% valid (able to be converted)
                # This last check is to account for single digit months.
                # 3/3/1995 will not be parsed to a string because standard formats require 03/03/1995
                # At least 15% of dates naturally have both month and day as 2 digits numbers
                dates.append(s)
        except: # noqa: E722
            # Very stupid code, but I have to do it...
            pass
    
    return dates

def date_removal(df:PolarsFrame) -> PolarsFrame:
    '''Removes all date columns from dataframe. This algorithm will try to infer if string column is date.'''

    remove_cols = date_inferral(df) 
    logger.info(f"The following columns are dropped because they are dates. {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

def null_inferral(df:PolarsFrame, threshold:float=0.5) -> list[str]:
    '''Infers columns that have more than threshold pct of null values. Threshold should be between 0 and 1.'''
    return (df.lazy().null_count().collect()/len(df)).transpose(include_header=True, column_names=["null_pct"])\
                    .filter(pl.col("null_pct") >= threshold)\
                    .get_column("column").to_list()

def null_removal(df:PolarsFrame, threshold:float=0.5) -> PolarsFrame:
    '''Removes columns with more than threshold pct of null values. Threshold should be between 0 and 1.'''

    remove_cols = null_inferral(df, threshold) 
    logger.info(f"The following columns are dropped because they have more than {threshold*100:.2f}%"
                f" null values. {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

def var_inferral(df:PolarsFrame, threshold:float, target:str) -> list[str]:
    '''Infers columns that have lower than threshold variance. Target will not be included.'''
    return df.lazy().select(
                pl.col(x).var() for x in get_numeric_cols(df) if x != target
            ).collect().transpose(include_header=True, column_names=["var"])\
            .filter(pl.col("var") < threshold).get_column("column").to_list() 

def var_removal(df:PolarsFrame, threshold:float, target:str) -> PolarsFrame:
    '''Removes features with low variance. Features with > threshold variance will be kept. 
        Threshold should be positive.'''

    remove_cols = var_inferral(df, threshold, target) 
    logger.info(f"The following columns are dropped because they have lower than {threshold} variance. {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

# Really this is just an alias
regex_inferral = get_cols_regex

def regex_removal(df:PolarsFrame, pattern:str, lowercase:bool=False) -> PolarsFrame:
    '''Remove columns if they satisfy some regex rules.'''
    remove_cols = get_cols_regex(df, pattern, lowercase)
    logger.info(f"The following columns are dropped because their names satisfy the regex rule: {pattern}."
                f" {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    
    return df.drop(remove_cols)

def get_unique_count(df:PolarsFrame) -> pl.DataFrame:
    '''Gets unique counts for columns.'''
    return df.lazy().select(
        pl.col(x).n_unique() for x in df.columns
    ).collect().transpose(include_header=True, column_names=["n_unique"])


# Really this is just an alias
def unique_inferral(df:PolarsFrame, threshold:float=0.9) -> list[str]:
    '''Infers columns that have higher than threshold pct of unique values.'''
    return get_unique_count(df).with_columns(
        (pl.col("n_unique")/len(df)).alias("unique_pct")
    ).filter(pl.col("unique_pct") >= threshold)\
    .get_column("column").to_list()

def unique_removal(df:PolarsFrame, threshold:float=0.9) -> PolarsFrame:
    '''Remove columns that have higher than threshold pct of unique values.'''

    remove_cols = unique_inferral(df, threshold)
    logger.info(f"The following columns are dropped because more than {threshold*100:.2f}% of unique values."
                f" {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

# Discrete = string or column that has < max_n_unique count of unique values or having unique_pct < threshold.
# Is this a good definition?
def discrete_inferral(df:PolarsFrame
    , threshold:float=0.1
    , max_n_unique:int=100
    , exclude:Optional[list[str]]=None
) -> list[str]:
    '''
        A column that satisfies either n_unique < max_n_unique or unique_pct < threshold 
        will be considered discrete.
    '''
    exclude_list = [] if exclude is None else exclude
    return get_unique_count(df).filter(
        ((pl.col("n_unique") < max_n_unique) | (pl.col("n_unique")/len(df) < threshold)) 
        & (~pl.col("column").is_in(exclude_list)) # is not in
    ).get_column("column").to_list()

def constant_inferral(df:PolarsFrame, include_null:bool=True) -> list[str]:
    temp = get_unique_count(df).filter(pl.col("n_unique") <= 2)

    remove_cols = temp.filter(pl.col("n_unique") == 1).get_column("column").to_list() 
    if include_null: # This step is kind of inefficient right now.
        binary = temp.filter(pl.col("n_unique") == 2).get_column("column")
        # df might be lazy. So we need to do things a bit differently.
        more = df.lazy().select(binary).select(
                pl.all().null_count()
            ).collect().transpose(include_header=True, column_names=["null_count"])\
                .filter(pl.col("null_count") > 0)\
                .get_column("column").to_list()
        
        remove_cols.extend(more)

    return remove_cols

def constant_removal(df:PolarsFrame, include_null:bool=True) -> PolarsFrame:
    '''Removes all constant columns from dataframe.
        Arguments:
            df:
            include_null: if true, then columns with two distinct values like [value_1, null] will be considered a 
                constant column.

        Returns: 
            the df without constant columns
    '''
    remove_cols = constant_inferral(df, include_null)
    logger.info(f"The following columns are dropped because they are constants. {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

def remove_if_exists(df:PolarsFrame, cols:list[str]) -> PolarsFrame:
    '''Removes the given columns if they exist in the dataframe.'''
    remove_cols = list(set(cols).intersection(set(df.columns)))
    logger.info(f"The following columns are dropped. {remove_cols}.\nRemoved a total of {len(remove_cols)} columns.")
    return df.drop(columns=remove_cols)