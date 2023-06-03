import polars as pl 
import re
from datetime import datetime 
from typing import Final, Any
from dataclasses import dataclass

_POLARS_NUMERICAL_TYPES:Final[list[pl.DataType]] = [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]

#----------------------------------------------------------------------------------------------#
# Generic columns checks                                                                       #
#----------------------------------------------------------------------------------------------#

def get_numeric_cols(df:pl.DataFrame, exclude:list[str]=None) -> list[str]:
    ''' 
    
    '''
    output = []
    exclude_list = [] if exclude is None else exclude
    for c,t in zip(df.columns, df.dtypes):
        if t in _POLARS_NUMERICAL_TYPES and c not in exclude_list:
            output.append(c)
    return output

def get_string_cols(df:pl.DataFrame, exclude:list[str]=None) -> list[str]:
    output = []
    exclude_list = [] if exclude is None else exclude
    for c,t in zip(df.columns, df.dtypes):
        if t == pl.Utf8 and c not in exclude_list:
            output.append(c)
    return output

def get_bool_cols(df:pl.DataFrame) -> list[str]:
    return [c for c,t in zip(df.columns, df.dtypes) if t == pl.Boolean]

def get_cols_regx(df:pl.DataFrame, pattern:str, lowercase:bool=False) -> list[str]:

    reg = re.compile(pattern)
    if lowercase:
        return [f for f in df.columns if reg.search(f)]
    return [f for f in df.columns if reg.search(f.lower())]

def dtype_mapping(d: Any) -> str: # dtype can be a "pl.datatype" or just some random data for which we want to infer a generic type.
    if isinstance(d, str) or d == pl.Utf8:
        return "string"
    elif isinstance(d, (int,float)) or d in _POLARS_NUMERICAL_TYPES:
        return "numeric"
    elif isinstance(d, bool) or d == pl.Boolean:
        return "bool"
    elif isinstance(d, datetime) or d in [pl.Datetime, pl.Date, pl.Time]:
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

    skew_and_kt = df.select(pl.col(c).skew() for c in df.columns).transpose(include_header=True, column_names=["skew"])\
                    .join(
                        df.select(pl.col(c).kurtosis() for c in df.columns).transpose(include_header=True, column_names=["kurtosis"])
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

def null_inferral(df:pl.DataFrame, threshold:float=0.5) -> list[str]:
    return (df.null_count()/len(df)).transpose(include_header=True, column_names=["null_pct"])\
                    .filter(pl.col("null_pct") > threshold)\
                    .get_column("column").to_list() 

def null_removal(df:pl.DataFrame, threshold:float=0.5) -> pl.DataFrame:
    '''
        Removes columns with more than threshold% null values.

        Arguments:
            df:
            threshold:

        Returns:
            df without null_pct > threshold columns
    '''

    remove_cols = null_inferral(df, threshold)    
    print(f"The following columns are dropped because they have more than {threshold*100:.2f}% null values. {remove_cols}.")
    print(f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

def var_inferral(df:pl.DataFrame, threshold:float, target:str) -> list[str]:
    var_expr = (pl.col(x).var() for x in get_numeric_cols(df) if x != target)
    return df.select(var_expr).transpose(include_header=True, column_names=["var"])\
                    .filter(pl.col("var") < threshold).get_column("column").to_list() 

def var_removal(df:pl.DataFrame, threshold:float, target:str) -> pl.DataFrame:
    '''
        Removes features with low variance. Features with > threshold variance will be kept. This only works for numerical columns.
        Note that this can effectively remove (numerical) constants as well. It is, however, hard to come up with
        a uniform threshold for all features, as numerical features can be at very different scales. 

        Arguments:
            df:
            threshold:
            target: target in your predictive model. Some targets may have low variance, e.g. imbalanced binary targets. So we should exclude it.

        Returns:
            df without columns with < threshold var.
    '''

    remove_cols = var_inferral(df, threshold, target)    
    print(f"The following numeric columns are dropped because they have lower than {threshold} variance. {remove_cols}")
    print(f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

# Really this is just an alias
regx_inferral = get_cols_regx

def regex_removal(df:pl.DataFrame, pattern:str, lowercase:bool=False) -> pl.DataFrame:
    remove_cols = get_cols_regx(df, pattern, lowercase)
    print(f"The following numeric columns are dropped because their names satisfy the regex rule: {pattern}. {remove_cols}")
    print(f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)


def get_unique_count(df:pl.DataFrame) -> pl.DataFrame:
    return df.select(
        (pl.col(x).n_unique() for x in df.columns)
    ).transpose(include_header=True, column_names=["n_unique"])

# Really this is just an alias
def unique_inferral(df:pl.DataFrame, threshold:float=0.9) -> list[str]:
    return get_unique_count(df).with_columns(
        (pl.col("n_unique")/len(df)).alias("unique_pct")
    ).filter(pl.col("unique_pct") > threshold).get_column("column").to_list()

def unique_removal(df:pl.DataFrame, threshold:float=0.9) -> pl.DataFrame:
    remove_cols = unique_inferral(df, threshold)
    print(f"The following columns are dropped because more than {threshold*100:.2f}% of values are unique. {remove_cols}")
    print(f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

def constant_inferral(df:pl.DataFrame, include_null:bool=True) -> list[str]:
    temp = get_unique_count(df).filter(pl.col("n_unique") <= 2)
    remove_cols = temp.filter(pl.col("n_unique") == 1).get_column("column").to_list() # These are constants, remove.
    if include_null: # This step is kind of inefficient right now.
        binary = temp.filter(pl.col("n_unique") == 2).get_column("column")
        for b in binary: 
            if df.get_column(b).null_count() > 0:
                remove_cols.append(b)
    return remove_cols

def constant_removal(df:pl.DataFrame, include_null:bool=True) -> pl.DataFrame:
    '''Removes all constant columns from dataframe.
        Arguments:
            df:
            include_null: if true, then columns with two distinct values like [value_1, null] will be considered a 
                constant column.

        Returns: 
            the df without constant columns
    '''
    remove_cols = constant_inferral(df, include_null)
    print(f"The following columns are dropped because they are constants. {remove_cols}.")
    print(f"Removed a total of {len(remove_cols)} columns.")
    return df.drop(remove_cols)

def remove_if_exists(df:pl.DataFrame, to_drop:list[str]) -> pl.DataFrame:
    drop = list(set(to_drop).intersection(set(df.columns)))
    return df.drop(columns=drop)