import polars as pl 
import numpy as np 
from enum import Enum
from dataclasses import dataclass
import json 
from .eda_prescreen import get_bool_cols, get_numeric_cols, get_string_cols, get_unique_count, dtype_mapping

@dataclass
class TransformationResult:
    transformed: pl.DataFrame
    mapping: pl.DataFrame

    # todo!
    def __str__(self):
        pass

    def __iter__(self):
        return iter((self.transformed, self.mapping))

# Oh FFF! How I miss Rust Enums
class ImputationStartegy(Enum):
    CONST = 1 
    MEDIAN = 2
    MEAN = 3
    MODE = 4

def impute(df:pl.DataFrame
        , cols_to_impute:list[str]
        , strategy:ImputationStartegy=ImputationStartegy.MEDIAN
        , const:int = 1) -> TransformationResult:
    
    '''
        Arguments:
            df:
            num_cols_to_impute:
            strategy:
            const: only uses this value if strategy = ImputationStartegy.CONST
    
    '''
    
    mapping = pl.from_records([cols_to_impute], schema=["feature"]).with_columns(
        pl.lit(str(strategy.name)).alias("imputation_strategy")
    )
    # Given Strategy, define expressions
    if strategy == ImputationStartegy.MEDIAN:
        all_medians = df[cols_to_impute].median().to_numpy().ravel()
        exprs = (pl.col(c).fill_null(all_medians[i]) for i,c in enumerate(cols_to_impute))
        values = pl.Series("impute_by", all_medians)
        mapping.insert_at_idx(2, values)      

    elif strategy == ImputationStartegy.MEAN:
        all_means = df[cols_to_impute].mean().to_numpy().ravel()
        exprs = (pl.col(c).fill_null(all_means[i]) for i,c in enumerate(cols_to_impute))
        values = pl.Series("impute_by", all_means)
        mapping.insert_at_idx(2, values)  

    elif strategy == ImputationStartegy.CONST:
        exprs = (pl.col(c).fill_null(const) for c in cols_to_impute)
        mapping.insert_at_idx(2, pl.Series("impute_by", [const]*len(cols_to_impute)))

    elif strategy == ImputationStartegy.MODE:
        all_modes = df.select(pl.col(c).mode() for c in cols_to_impute).to_numpy().ravel()
        exprs = (pl.col(c).fill_null(all_modes[i]) for i,c in enumerate(cols_to_impute))
        values = pl.Series("impute_by", all_modes)
        mapping.insert_at_idx(2, values)  

    else:
        print("This shouldn't happen.")
        # Return None or not? 

    transformed = df.with_columns(exprs)
    
    return TransformationResult(transformed=transformed, mapping=mapping)

class ScalingStrategy(Enum):
    NORMALIZE = 1
    MIN_MAX = 2
    CONST = 3

def scale(df:pl.DataFrame
        , num_cols_to_scale:list[str]
        , strategy:ScalingStrategy=ScalingStrategy.NORMALIZE
        , const:int = 1) -> TransformationResult:
    
    '''
        Arguments:
            df:
            num_cols_to_impute:
            strategy:
            const: only uses this value if strategy = ImputationStartegy.CONST
    
    '''
    
    mapping = pl.from_records([num_cols_to_scale], schema=["feature"]).with_columns(
        pl.lit(str(strategy.name)).alias("scaling_strategy")
    )

    if strategy == ScalingStrategy.NORMALIZE:
        all_means = df[num_cols_to_scale].mean().to_numpy().ravel()
        all_stds = df[num_cols_to_scale].std().to_numpy().ravel()
        exprs = (((pl.col(c) - pl.lit(all_means[i]))/(pl.lit(all_stds[i])) for i,c in enumerate(num_cols_to_scale)))
        args = (json.dumps({"mean":m, "std":s}) for m,s in zip(all_means, all_stds))
        mapping.insert_at_idx(2, pl.Series("params", args))

    elif strategy == ScalingStrategy.MIN_MAX:
        all_maxs = df[num_cols_to_scale].max().to_numpy().ravel()
        all_mins = df[num_cols_to_scale].min().to_numpy().ravel()
        exprs = ((pl.col(c) - pl.lit(all_mins[i]))/(pl.lit(all_maxs[i] - all_mins[i])) for i,c in enumerate(num_cols_to_scale))
        args = (json.dumps({"min":m, "max":mm}) for m, mm in zip(all_mins, all_maxs))
        mapping.insert_at_idx(2, pl.Series("params", args))

    elif strategy == ScalingStrategy.CONST:
        exprs = (pl.col(c)/const for c in num_cols_to_scale)
        mapping.insert_at_idx(2, pl.Series("params", pl.lit(str(const))))

    else:
        print("This shouldn't happen.")

    transformed = df.with_columns(exprs)
    
    return TransformationResult(transformed=transformed, mapping=mapping)

def boolean_transform(df:pl.DataFrame, keep_null:bool=True) -> pl.DataFrame:
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

def one_hot_encode(df:pl.DataFrame, one_hot_columns:list[str], separator:str="_") -> TransformationResult:
    '''

    
    '''
    res = df.to_dummies(columns=one_hot_columns, separator=separator)
    records = []
    for c in one_hot_columns:
        for cc in filter(lambda name: c in name, res.columns):
            records.append((c,cc))
    mapping = pl.from_records(records, orient="row", schema=["feature", "one_hot_derived"])
    return TransformationResult(transformed = res, mapping = mapping)

def fixed_sized_encode(df:pl.DataFrame, num_cols:list[str], bin_size:int=50) -> TransformationResult:
    '''Given a continuous variable, take the smallest `bin_size` of them, and call them bin 1, take the next
    smallest `bin_size` of them and call them bin 2, etc...
    
    '''
    pass

# Try to generalize this.
def percentile_encode(df:pl.DataFrame, num_cols:list[str]=None, exclude:list[str]=None) -> TransformationResult:
    '''
        Bin your continuous variable X into X_percentiles. This will create at most 100 + 1 bins, where each percentile could
        potentially be a bin and null will be mapped to bin = 0. Bin 1 means percentile 0 to 1. Generally, bin X groups the
        population from bin X-1 to bin X into one bucket.

        I see some potential optimization opportunities here.

        Arguments:
            df:
            num_cols: 
            exclude:

        Returns:
            (A transformed dataframe, a mapping table (value to percentile))
    
    '''

    # Percentile Binning

    num_list:list[str] = []
    exclude_list:list[str] = [] if exclude is None else exclude
    if isinstance(num_cols, list):
        num_list.extend(num_cols)
    else:
        num_list.extend(get_numeric_cols(df, exclude=exclude_list))

    tables:list[pl.DataFrame] = []
    exprs:list[pl.Expr] = []
    rename_dict:dict[str,str] = {}
    for c in num_list:
        percentile = df.groupby(c).agg(pl.count().alias("cnt"))\
            .sort(c)\
            .with_columns(
                ((pl.col("cnt").cumsum()*100)/len(df)).ceil().alias("percentile")
            ).groupby("percentile")\
            .agg((
                pl.col(c).min().alias("min"),
                pl.col(c).max().alias("max"),
                pl.col("cnt").sum().alias("cnt"),
            )).sort("percentile").select((
                pl.lit(c).alias("feature"),
                pl.col("percentile").cast(pl.UInt8),
                "min",
                "max",
                "cnt",
            ))
        
        first_row = percentile.select(["percentile","min", "max"]).to_numpy()[0, :] # First row
        # Need to handle an extreme case when percentile looks like 
        # percentile   min   max
        #  p1         null  null
        #  p2          ...   ...
        # This happens when there are so many nulls in the column.
        if np.isnan(first_row[2]):
            # Discard the first row if this is the case. 
            percentile = percentile.slice(1, length = None)

        temp_df = df.lazy().filter(pl.col(c).is_not_null()).sort(c).set_sorted(c)\
            .join_asof(other=percentile.lazy().set_sorted("max"), left_on=c, right_on="max", strategy="forward")\
            .select((c, "percentile"))\
            .unique().collect()
        
        mapping = dict(zip(temp_df[c], temp_df["percentile"]))

        exprs.append(
            pl.col(c).map_dict(mapping, default=0).cast(pl.UInt8)
        )
        rename_dict[c] = c + "_percentile"
        percentile = percentile.with_columns((
            pl.col("min").cast(pl.Float32),
            pl.col("max").cast(pl.Float32),
            pl.col("cnt").cast(pl.UInt32)
        )) # Need to do this because we need a uniform format in order to stack these columns.
        tables.append(percentile)

    res = df.with_columns(exprs).rename(rename_dict)
    mapping = pl.concat(tables)
    return TransformationResult(transformed=res, mapping=mapping)

def binary_encode(df:pl.DataFrame, binary_cols:list[str]=None, exclude:list[str]=None) -> TransformationResult:
    '''Encode the given columns as binary values.

        The goal of this function is to map binary categorical values into [0, 1], therefore reducing the amount of encoding
        you will have to do later. This is important when you want to keep feature dimension low and when you have many binary categorical
        variables. The values will be mapped to [0, 1] by the following rule:
            if value_1 < value_2, value_1 --> 0, value_2 --> 1. E.g. 'N' < 'Y' ==> 'N' --> 0 and 'Y' --> 1
        
        In case the two distinct values are [None, value_1], and you decide to treat this variable as a binary category, then
        None --> 0 and value_1 --> 1. (If you apply constant_removal first then this column will be seen as constant and dropped.)
        
        Using one-hot-encoding will map binary categorical values to 2 columns (except when you specify drop_first=True in pd.get_dummies),
        therefore introducing unnecessary dimension. So it is better to prevent it.

        In case the distinct values the column are [null, value_1, value_2], you must first impute this column if you want this method
        to count this as a binary column. 

        Arguments:
            df:
            binary_cols: the binary_cols you wish to convert. If no input, will infer (might take time because counting unique values for each column is not cheap).
            exclude: the columns you wish to exclude in this transformation. (Only applies if you are letting the system auto-detecting binary columns.)

        Returns: 
            (the transformed dataframe, mapping table between old values to [0,1])
    '''
    mapping = {"feature":[], "to_0":[], "to_1":[], "dtype":[]}
    exprs = []
    binary_list = []
    if isinstance(binary_cols, list):
        binary_list.extend(binary_cols)
    else:
        binary_columns = get_unique_count(df).filter((pl.col("n_unique") == 2) & (~pl.col("column").is_in(exclude))).get_column("column")
        binary_list.extend(binary_columns)     
    
    # Doing some repetitive operations here, but I am not sure how I can get all the data in one go.
    for b in binary_list:
        vals = df.get_column(b).unique().to_list()
        print(f"Transforming {b} into a binary column with [0, 1] ...")
        if len(vals) != 2:
            print(f"Found {b} has {len(vals)} unique values instead of 2. Not a binary variable. Ignored.")
            continue
        if vals[0] is None: # Weird code, but we need this case.
            pass
        elif vals[1] is None:
            vals[0], vals[1] = vals[1], vals[0]
        else:
            vals.sort()

        mapping["feature"].append(b)
        mapping["to_0"].append(vals[0] if vals[0] is None else str(vals[0])) # have to cast to str to avoid mixed types
        mapping["to_1"].append(str(vals[1])) # vals[1] is gauranteed to be not None by above logic
        mapping["dtype"].append(dtype_mapping(vals[1]))
        
        exprs.append(
            pl.when(pl.col(b).is_null()).then(0).otherwise(
                pl.when(pl.col(b) < vals[1]).then(0).otherwise(1)
            ).cast(pl.UInt8).alias(b) 
        )

    res = df.with_columns(exprs)
    mapping = pl.from_dict(mapping)
    return TransformationResult(transformed = res, mapping = mapping)

def get_ordinal_mapping_table(ordinal_mapping:dict[str, dict[str,int]]) -> pl.DataFrame:
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
        ).select(("feature", "value", "mapped_to"))
        mapping_tables.append(
            table
        )

    return pl.concat(mapping_tables)

def ordinal_auto_encode(df:pl.DataFrame, ordinal_cols:list[str]=None, exclude:list[str]=None) -> TransformationResult:
    '''
        Automatically applies ordinal encoding to the provided columns by the following logic:
            Sort the column, smallest value will be assigned to 0, second smallest will be assigned to 1...

        This will automatically detect string columns and apply this operation if ordinal_cols is not provided. 
        This method is great for string columns like age ranges, with values like ["10-20", "20-30"], etc...
        
        Arguments:
            df:
            ordinal_cols:
            exclude: the columns you wish to exclude in this transformation. (Only applies if you are letting the system auto-detecting binary columns.)
        
        Returns:
            (encoded df, mapping table)
    '''
    ordinal_list:list[str] = []
    if isinstance(ordinal_cols, list):
        ordinal_list.extend(ordinal_cols)
    else:
        ordinal_list.extend(get_string_cols(df, exclude=exclude))

    exprs:list[pl.Expr] = []
    ordinal_mapping:dict[str, dict[str,int]] = {}
    rename_dict:dict[str, str] = {}
    for c in ordinal_list:
        sorted_uniques = df.get_column(c).unique().sort()
        count = len(sorted_uniques)
        mapping:dict[str, int] = dict(zip(sorted_uniques, range(count)))
        ordinal_mapping[c] = mapping
        exprs.append(pl.col(c).map_dict(mapping).cast(pl.UInt32))
        rename_dict[c] = c + "_ordinal"

    res = df.with_columns(exprs).rename(rename_dict)
    mapping = get_ordinal_mapping_table(ordinal_mapping)
    return TransformationResult(transformed=res, mapping=mapping)

def ordinal_encode(df:pl.DataFrame, ordinal_mapping:dict[str, dict[str,int]], default:int = -1) -> TransformationResult:
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
    
    rename_dict:dict[str, str] = {}
    exprs:list[pl.Expr] = []
    for c in ordinal_mapping:
        if c in df.columns:
            mapping = ordinal_mapping[c]
            exprs.append(pl.col(c).map_dict(mapping, default=default).cast(pl.UInt32))
            rename_dict[c] = c + "_ordinal"
        else:
            print(f"Found that column {c} is not in df. Skipped.")

    res = df.with_columns(exprs).rename(rename_dict)
    mapping = get_ordinal_mapping_table(ordinal_mapping)
    return TransformationResult(transformed=res, mapping=mapping)

def smooth_target_encode(df:pl.DataFrame, target:str
                         , smoothing_factor:float
                         , min_samples_leaf:int
                         , cat_cols:list[str]=None) -> TransformationResult:
    
    '''Smooth target encoding for binary classification.

        See https://towardsdatascience.com/dealing-with-categorical-variables-by-using-target-encoder-a0f1733a4c69

        Arguments:
            df:
            target:
            smoothing_factor:
            min_samples_leaf:
            cat_cols
    
    '''
    
    # Only works for binary target for now 
    cat_list:list[str] = []
    if isinstance(cat_cols, list):
        cat_list.extend(cat_cols)
    else:
        cat_list.extend(get_string_cols(df))

    # Check if it is binary or not.
    target_uniques = list(df.get_column(target).unique().sort())
    if target_uniques != [0,1]:
        raise ValueError(f"The target column {target} must be a binary target with 0 and 1 representing the two classes.")

    p = df.get_column(target).mean() # probability of target = 1
    all_refs:list[pl.DataFrame] = []
    exprs:list[pl.Expr] = []
    for c in cat_list:
        ref = df.groupby(c).agg((
            pl.col(target).sum().alias("cnt")
        )).with_columns(
            (pl.col("cnt")/len(df)).alias("cond_p")
        ).with_columns(
            (1 / (1 + ((-(pl.col("cnt") - pl.lit(min_samples_leaf)))/pl.lit(smoothing_factor)).exp())).alias("alpha")
        ).with_columns(
            (pl.col("alpha") * pl.col("cond_p") + (pl.lit(1) - pl.col("alpha")) * pl.lit(p)).alias("encoded_as")
        ).select((c, "encoded_as"))
        
        all_refs.append(ref)
        local_map = dict(zip(ref[c], ref["encoded_as"]))
        exprs.append(pl.col(c).map_dict(local_map))
        
    res = df.with_columns(exprs)
    mapping = pl.concat(all_refs)

    return TransformationResult(transformed=res, mapping=mapping) 