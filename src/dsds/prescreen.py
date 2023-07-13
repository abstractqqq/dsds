from .type_alias import (
    PolarsFrame
    , KSAlternatives
    , CommonContinuousDist
    , CPU_COUNT
    , POLARS_DATETIME_TYPES
    , POLARS_NUMERICAL_TYPES
)
from .sample import (
    lazy_sample
)
from .blueprint import(
    Blueprint
)

import polars.selectors as cs
import polars as pl
import logging  
from datetime import datetime 
from typing import Any, Optional, Tuple
from itertools import combinations
from scipy.stats import (
    ks_2samp
    , kstest
)
from concurrent.futures import as_completed, ThreadPoolExecutor
from tqdm import tqdm
from math import comb

logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------------#
# Generic columns checks | Only works with Polars because Pandas's data types suck!            #
#----------------------------------------------------------------------------------------------#

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

def get_cols_regex(df:PolarsFrame, pattern:str) -> list[str]:
    '''Returns columns that have names matching the regex pattern.'''
    return df.select(cs.matches(pattern=pattern)).columns

def lowercase_columns(df:PolarsFrame, persist:bool=False) -> PolarsFrame:
    '''
    Lowercases all column names.

    Set persist = True if this needs to be remembered by the blueprint.
    '''
    output = df.rename({c: c.lower() for c in df.columns})
    if isinstance(df, pl.LazyFrame) and persist:
        return output.blueprint.apply_func(df, lowercase_columns, kwargs = {})
    return output

def drop_nulls(
    df:PolarsFrame
    , subset:Optional[list[str]] = None
    , persist:bool=False
) -> PolarsFrame:
    '''
    A wrapper function for Polars' drop_nulls so that it can be used in pipeline.

    Set persist = True if this needs to be remembered by the blueprint.
    '''
    if subset is None:
        by = df.columns
    else:
        by = subset

    if isinstance(df, pl.LazyFrame):
        cond = pl.lit(True)
        for c in by:
            cond = cond & pl.col(c).is_not_null()
        
        if persist:
            return df.blueprint.filter(cond)
        return df.filter(cond)
    else:
        return df.drop_nulls(subset=by)

def filter(
    df:PolarsFrame
    , condition: pl.Expr
    , persist: bool = False
) -> PolarsFrame:
    ''' 
    A wrapper function for Polars' filter so that it can be used in pipeline.

    Set persist = True if this needs to be remembered by the blueprint.
    '''
    if isinstance(df, pl.LazyFrame):
        if persist:
            return df.blueprint.filter(condition)
    return df.filter(condition)

def check_binary_target(df:PolarsFrame, target:str) -> bool:
    '''Checks if target is binary or not. Only binary targets with 0s and 1s will pass.'''
    target_uniques = df.lazy().select(pl.col(target).unique()).collect().get_column(target)
    if len(target_uniques) != 2:
        logger.error("Target is not binary.")
        return False
    elif not (0 in target_uniques and 1 in target_uniques):
        logger.error("The binary target is not encoded as 0s and 1s.")
        return False
    return True
    
def check_target_cardinality(df:PolarsFrame, target:str) -> pl.DataFrame:
    '''Returns a dataframe showing the cardinality of different target values.'''
    return df.lazy().groupby(target).count().sort(target).collect()

def check_columns_types(df:PolarsFrame, cols:Optional[list[str]]=None) -> str:
    '''Returns the unique types of given columns in a single string. If multiple types are present
    they are joined by a |. If cols is not given, automatically uses all columns.'''
    if cols is None:
        check_cols:list[str] = df.columns
    else:
        check_cols:list[str] = cols 

    types = set(dtype_mapping(t) for t in df.select(check_cols).dtypes)
    return "|".join(types) if len(types) > 0 else "unknown"
    
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

# Add a slim option that returns fewer stats? This is generic describe.
# Separate str and numeric?
def describe(
    df:PolarsFrame
    , sample_frac:float = 0.75
) -> pl.DataFrame:
    '''
    Profile the data.

    '''

    if isinstance(df, pl.LazyFrame):
        df_local = lazy_sample(df, sample_frac=sample_frac)
    else:
        df_local = df
    
    temp = df_local.describe()
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
    # Combine all
    nums = ("count", "null_count", "mean", "std", "median", "25%", "75%")
    final = temp.transpose(include_header=True, column_names=desc).with_columns(
        pl.col(c).cast(pl.Float64) for c in nums
    ).with_columns(
        null_pct = pl.col("null_count")/pl.col("count")
        , CoV = pl.col("std") / pl.col("mean")
        , dtype = pl.col("column").map_dict(dtypes_dict)
    ).join(unique_counts, on="column").join(skew_and_kt, on="column")
    
    return final.select('column','count','null_count','null_pct','n_unique'
                        , 'unique_pct','mean','std', 'CoV','min','max','25%'
                        , 'median','75%', "skew", "kurtosis",'dtype')

# Numeric only describe. Be more detailed.

# String only describe. Be more detailed about interesting string stats.

def describe_str(df:PolarsFrame
    , words_to_count:Optional[list[str]]=None
    , sample_frac:float = 0.75
) -> pl.DataFrame:
    '''Gives some statistics about the string columns. Optionally you may pass a list
    of strings to compute the total occurrances of each of the words in the string columns. If input is a LazyFrame, 
    a sample of sample_pct will be used, and sample_pct will only be used in the lazy case. 

    '''
    strs = get_string_cols(df)
    df_str = df.select(strs)
    if isinstance(df, pl.LazyFrame):
        df_str = lazy_sample(df_str, sample_frac=sample_frac).collect()

    nstrs = len(strs)
    stats = df.select(strs).select(
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
    ).row(0)
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
            t = df_str.select(pl.all().str.count_match(w).sum().prefix("wc:")).row(0)
            output["total_"+ w + "_count"] = t

    return pl.from_dict(output)

# -----------------------------------------------------------------------------------------------
def drop(df:PolarsFrame, to_drop:list[str]) -> PolarsFrame:
    if isinstance(df, pl.LazyFrame):
        return df.blueprint.drop(to_drop)
    return df.drop(to_drop)

def non_numeric_removal(df:PolarsFrame, include_bools:bool=True) -> PolarsFrame:
    '''Removes all non-numeric columns. If include_bools = True, then keep boolean columns.'''
    
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

def pattern_inferral(
    df: PolarsFrame
    , pattern:str
    , sample_frac:float = 0.75
    , sample_count:int = 100_000
    , sample_rounds:int = 3
    , threshold:float = 0.9
    , count_null:bool = True
) -> list[str]:
    '''
    Find all string columns whose elements reasonably match the given pattern. The match logic can 
    be tuned using the all the parameters.

    Arguments:
        sample_frac: the pct of the total dataframe to use as basis
        sample_count: from the basis, how many rows to sample for each round 
        sample_rounds: how many rounds of sampling we are doing
        threhold: For each round, what is the match% that is needed to be a counted as a success. For instance, 
        in round 1, for column x, we have 92% match rate, and threshold = 0.9. We count column x as a match for 
        this round. In the end, the column must match for every round to be considered a real match.
        count_null: for individual matches, do we want to count null as a match or not? If the column has high null pct,
        the non-null values might mostly match the pattern. In this case, using count_null = True will match the column, 
        while count_null = False will most likely exclude the column.

    Returns:
        a list of columns that pass the matching test
    
    '''
    
    strs = get_string_cols(df)
    df_local = lazy_sample(df.lazy(), sample_frac=sample_frac).collect()    
    matches:set[str] = set(strs)
    sample_size = min(sample_count, len(df_local))
    for _ in range(sample_rounds):
        df_sample = df_local.sample(n = sample_size)
        fail = df_sample.select(
            (pl.when(pl.col(s).is_null()).then(count_null).otherwise(
                pl.col(s).str.contains(pattern)
            ).sum()/sample_size).alias(s) for s in strs
        ).transpose(include_header=True, column_names=["pattern_match_pct"])\
        .filter(pl.col("pattern_match_pct") < threshold).get_column("column")
        # If the match failes in this round, remove the column.
        matches.difference_update(fail)

    return list(matches)

def pattern_removal(
    df: PolarsFrame
    , pattern:str
    , sample_pct:float = 0.75
    , sample_count:int = 100_000
    , sample_rounds:int = 3
    , threshold:float = 0.9
    , count_null:bool = True
) -> PolarsFrame:
    
    remove_cols = pattern_inferral(
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

def email_inferral(
    df: PolarsFrame
    , sample_pct:float = 0.75
    , sample_count:int = 100_000
    , sample_rounds:int = 3
    , threshold:float = 0.9
    , count_null:bool = True
) -> list[str]:
    # Why does this regex not work?
    # r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+'
    return pattern_inferral(
        df
        , r'\S+@\S+\.\S+'
        , sample_pct
        , sample_count
        , sample_rounds
        , threshold 
        , count_null
    )

def email_removal(
    df: PolarsFrame
    , sample_pct:float = 0.75
    , sample_count:int = 100_000
    , sample_rounds:int = 3
    , threshold:float = 0.9
    , count_null:bool = True
) -> PolarsFrame:
    
    emails = email_inferral(df, sample_pct, sample_count, sample_rounds, threshold, count_null)
    logger.info(f"The following columns are dropped because they are emails. {emails}.\n"
            f"Removed a total of {len(emails)} columns.")
    
    return drop(df, emails)

# Check for columns that are US zip codes.
# Might add options for other countries later.
def zipcode_inferral():
    # Match string using pattern inferral
    # Take a look at integers too, are they always 5 digits? 
    pass

def date_inferral(df:PolarsFrame) -> list[str]:
    '''Infers date columns in dataframe. This inferral is not perfect.'''
    logger.info("Date Inferral is error prone due to the huge variety of date formats. Please use with caution.")
    
    dates = [c for c,t in zip(df.columns, df.dtypes) if t in POLARS_DATETIME_TYPES]
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

def date_removal(df:PolarsFrame) -> PolarsFrame:
    '''Removes all date columns from dataframe. This algorithm will try to infer if string column is date.'''

    remove_cols = date_inferral(df) 
    logger.info(f"The following columns are dropped because they are dates. {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    return drop(df, remove_cols)

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
    return drop(df, remove_cols)

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
    return drop(df, remove_cols)

# Really this is just an alias
regex_inferral = get_cols_regex

def regex_removal(df:PolarsFrame, pattern:str, lowercase:bool=False) -> PolarsFrame:
    '''Remove columns if they satisfy some regex rules.'''
    remove_cols = get_cols_regex(df, pattern, lowercase)
    logger.info(f"The following columns are dropped because their names satisfy the regex rule: {pattern}."
                f" {remove_cols}.\n"
                f"Removed a total of {len(remove_cols)} columns.")
    
    return drop(df, remove_cols)

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
    return drop(df, remove_cols)

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

def conti_inferral(
    df:PolarsFrame
    , discrete_threshold:float = 0.1
    , discrete_max_n_unique:int = 100
    , exclude:Optional[list[str]]=None
) -> list[str]:
    exclude_list = [] if exclude is None else exclude
    discrete = discrete_inferral(df, discrete_threshold, discrete_max_n_unique)
    return df.select(cs.numeric() & ~cs.by_name(exclude_list) & ~cs.by_name(discrete)).columns

def constant_inferral(df:PolarsFrame, include_null:bool=True) -> list[str]:
    temp = get_unique_count(df).filter(pl.col("n_unique") <= 2)
    remove_cols = temp.filter(pl.col("n_unique") == 1).get_column("column").to_list() 
    if include_null: # This step is kind of inefficient right now.
        binary = temp.filter(pl.col("n_unique") == 2).get_column("column")
        nc = df.lazy().select(binary).null_count().collect().row(0)
        remove_cols.extend(
            binary[i] for i in range(len(nc)) if nc[i] > 0
        )

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
    return drop(df, remove_cols)

def remove_if_exists(df:PolarsFrame, cols:list[str]) -> PolarsFrame:
    '''Removes the given columns if they exist in the dataframe.'''
    remove_cols = list(set(cols).intersection(set(df.columns)))
    logger.info(f"The following columns are dropped. {remove_cols}.\nRemoved a total of {len(remove_cols)} columns.")
    return drop(df, remove_cols)

#----------------------------------------------------------------------------------------------#
# More advanced Methods
#----------------------------------------------------------------------------------------------#

def _ks_compare(
    df:pl.DataFrame
    , pair:Tuple[str, str]
    , alt:KSAlternatives="two-sided"
) -> Tuple[Tuple[str, str], float, float]:
    
    res = ks_2samp(df.get_column(pair[0]), df.get_column(pair[1]), alt)
    return (pair, res.statistic, res.pvalue)

def ks_compare(
    df:PolarsFrame
    , target:Optional[str] = None
    , smaple_frac:float = 0.75
    , test_cols:Optional[list[str]] = None
    , alt: KSAlternatives = "two-sided"
    , skip:int = 0
    , max_comp:int = 1000
) -> pl.DataFrame:
    '''Run ks-stats on all non-discrete columns in the dataframe. If test_cols is None, it will infer non-discrete 
    continuous columns. See docstring of discrete_inferral to see what is considered discrete. Provide the target 
    so that it will not be included in the comparisons. Since ks 2 sample comparison is relatively expensive, we will
    always sample 75% of the dataset, unless the user specifies a different sample_frac.

    Note: this will only run on all 2 combinations of columns, starting from skip and end at 
    skip + max_comp.

    Note: The null hypothesis is that the two columns come from the same distribution. Therefore a small p-value means
    that they do not come from the same distribution. Having p-value > threshold does not mean they have the same 
    distribution automatically, and it requires more examination to reach the conclusion.
    '''
    if test_cols is None:
        nums = [f for f in get_numeric_cols(df) if f not in discrete_inferral(df)]
    else:
        nums = test_cols

    if target in nums:
        nums.remove(target)
    sorted(nums)
    if isinstance(df, pl.LazyFrame):
        df_test = lazy_sample(df.select(nums).lazy(), sample_frac=smaple_frac).collect()
    else:
        df_test = df.select(nums).sample(fraction=smaple_frac)

    n_c2 = comb(len(nums), 2)
    last_index = min(skip + max_comp, n_c2)
    results = []
    to_test = enumerate(combinations(nums, 2), start=skip)
    pbar = tqdm(total=min(max_comp, n_c2 - skip), desc="Comparisons")
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as ex:
        for f in as_completed(ex.submit(_ks_compare, df_test, p, alt) for i, p in to_test if i < last_index):
            results.append(f.result())
            pbar.update(1)

    pbar.close()
    return pl.from_records(results, schema=["combination", "ks-stats", "p-value"])

def _dist_inferral(df:pl.DataFrame, c:str, dist:CommonContinuousDist) -> Tuple[str, float, float]:
    res = kstest(df[c], dist)
    return (c, res.statistic, res.pvalue)

def dist_test(
    df: PolarsFrame
    , which_dist:CommonContinuousDist
    , smaple_frac:float = 0.75
    , target: Optional[str] = None
) -> pl.DataFrame:
    '''Tests if the numeric columns follow the given distribution by using the KS test. If
    target is provided it will be excluded. The null hypothesis is that the columns follow the given distribution. 
    We sample 75% of data because ks test is relatively expensive.
    '''
    
    nums = get_numeric_cols(df, exclude=[target])
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
    return pl.from_records(results, schema=["feature", "ks-stats", "p-value"])

def suggest_normal(
    df:PolarsFrame
    , target: Optional[str] = None
    , threshold:float = 0.05
) -> list[str]:
    '''Suggests which columns are normally distributed. This takes the columns for which the null hypothesis
    cannot be rejected in the dist_test (KS test).
    '''
    return dist_test(df, "norm", target=target).filter(pl.col("p-value") > threshold)\
        .get_column("feature").to_list()

def suggest_uniform(
    df:PolarsFrame
    , target: Optional[str] = None
    , threshold:float = 0.05
) -> list[str]:
    '''Suggests which columns are uniformly distributed. This takes the columns for which the null hypothesis
    cannot be rejected in the dist_test (KS test).
    '''
    return dist_test(df, "uniform", target=target).filter(pl.col("p-value") > threshold)\
        .get_column("feature").to_list()

def suggest_lognormal(
    df:PolarsFrame
    , target: Optional[str] = None
    , threshold:float = 0.05
) -> list[str]:
    '''Suggests which columns are log-normally distributed. This takes the columns which the null hypothesis
    cannot be rejected in the dist_test (KS test).
    '''
    return dist_test(df, "lognorm", target=target).filter(pl.col("p-value") > threshold)\
        .get_column("feature").to_list()

def suggest_dist(
    df:PolarsFrame
    , target: Optional[str] = None
    , threshold:float = 0.05
    , dist: CommonContinuousDist = "norm"
) -> list[str]:
    '''Suggests which columns follow the given distribution. This returns the columns which the null hypothesis
    cannot be rejected in the dist_test (KS test).
    '''
    return dist_test(df, dist, target=target).filter(pl.col("p-value") > threshold)\
        .get_column("feature").to_list()