from __future__ import annotations

import orjson
import polars as pl
import numpy as np 
from enum import Enum
from typing import Any
from dataclasses import dataclass
from .eda_prescreen import get_bool_cols, get_numeric_cols, get_string_cols, get_unique_count, dtype_mapping

class ImputationStartegy(Enum):
    CONST = "CONST"
    MEDIAN = 'MEDIAN'
    MEAN = "MEAN"
    MODE = "MODE"

class ScalingStrategy(Enum):
    NORMALIZE = "NORMALIZE"
    MIN_MAX = "MIN-MAX"
    CONST = "CONST"

class EncodingStrategy(Enum):
    ORDINAL = "ORDINAL"
    ORDINAL_AUTO = "ORDINAL_AUTO"
    TARGET = "TARGET"
    ONE_HOT = "ONE-HOT"
    BINARY = "BINARY"
    PERCENTILE = "PERCENTILE"

@dataclass
class ImputationRecord:
    features:list[str]
    strategy:ImputationStartegy
    values:list[float] | np.ndarray

    def __iter__(self):
        return zip(self.features, [self.strategy]*len(self.features), self.values)
    
    def materialize(self) -> pl.DataFrame:
        return pl.from_records(list(self), schema=["feature", "imputation_strategy", "value_used"])
    
    def __str__(self) -> str:
        return orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY).decode()

@dataclass
class ScalingRecord:
    features:list[str]
    strategy:ScalingStrategy
    values:list[dict[str, float]]

    def __iter__(self):
        vals = (orjson.dumps(v, option=orjson.OPT_SERIALIZE_NUMPY).decode() for v in self.values)
        return zip(self.features, [self.strategy]*len(self.features), vals)
    
    def materialize(self) -> pl.DataFrame:
        return pl.from_records(list(self), schema=["feature", "scaling_strategy", "scaling_meta_data"])
    
    def __str__(self) -> str:
        return orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY).decode()

@dataclass
class EncoderRecord:
    features:list[str]
    strategy:EncodingStrategy
    mappings:list[dict[Any, Any]]

    def __iter__(self):
        vals = (orjson.dumps(v, option=orjson.OPT_SERIALIZE_NUMPY|orjson.OPT_NON_STR_KEYS).decode() for v in self.mappings)
        return zip(self.features, [self.strategy]*len(self.features), vals)
    
    def materialize(self) -> pl.DataFrame:
        return pl.from_records(list(self), schema=["feature", "encoding_strategy", "maps"])
    
    def __str__(self) -> str:
        return orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY|orjson.OPT_NON_STR_KEYS).decode()

@dataclass
class TransformationResult:
    transformed: pl.DataFrame
    mapping: pl.DataFrame | ImputationRecord | ScalingRecord | EncoderRecord

    def __iter__(self):
        return iter((self.transformed, self.mapping))
    
    def materialize(self) -> pl.DataFrame:
        return self.get_mapping_table()

    def get_mapping_table(self) -> pl.DataFrame:
        if isinstance(self.mapping, pl.DataFrame):
            return self.mapping
        else:
            return self.mapping.materialize()

def impute(df:pl.DataFrame
    , cols:list[str]
    , strategy:ImputationStartegy=ImputationStartegy.MEDIAN
    , const:int = 1) -> TransformationResult:
    
    '''
        Arguments:
            df:
            cols:
            strategy:
            const: only uses this value if strategy = ImputationStartegy.CONST
    
    '''

    # Given Strategy, define expressions
    if strategy == ImputationStartegy.MEDIAN:
        all_medians = df[cols].median().to_numpy().ravel()
        exprs = (pl.col(c).fill_null(all_medians[i]) for i,c in enumerate(cols))
        impute_record = ImputationRecord(cols, strategy, all_medians)

    elif strategy == ImputationStartegy.MEAN:
        all_means = df[cols].mean().to_numpy().ravel()
        exprs = (pl.col(c).fill_null(all_means[i]) for i,c in enumerate(cols))
        impute_record = ImputationRecord(cols, strategy, all_means)

    elif strategy == ImputationStartegy.CONST:
        exprs = (pl.col(c).fill_null(const) for c in cols)
        impute_record = ImputationRecord(cols, strategy, [const]*len(cols))

    elif strategy == ImputationStartegy.MODE:
        all_modes = df.select(pl.col(c).mode() for c in cols).to_numpy().ravel()
        exprs = (pl.col(c).fill_null(all_modes[i]) for i,c in enumerate(cols))
        impute_record = ImputationRecord(cols, strategy, all_modes)

    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    transformed = df.with_columns(exprs)
    return TransformationResult(transformed=transformed, mapping=impute_record)

def impute_by(df:pl.DataFrame, rec:ImputationRecord) -> pl.DataFrame:
    return df.with_columns(
        pl.col(f).fill_null(v) for f, v in zip(rec.features, rec.values)
    )

def scale(df:pl.DataFrame
    , cols:list[str]
    , strategy:ScalingStrategy=ScalingStrategy.NORMALIZE
    , const:int = 1) -> TransformationResult:
    
    '''
        Arguments:
            df:
            cols:
            strategy:
            const: only uses this value if strategy = ImputationStartegy.CONST
    
    '''
    
    if strategy == ScalingStrategy.NORMALIZE:
        all_means = df[cols].mean().to_numpy().ravel()
        all_stds = df[cols].std().to_numpy().ravel()
        exprs = (((pl.col(c) - pl.lit(all_means[i]))/(pl.lit(all_stds[i])) for i,c in enumerate(cols)))
        scale_data = [{"mean":m, "std":s} for m,s in zip(all_means, all_stds)]
        scaling_records = ScalingRecord(cols, strategy, scale_data)

    elif strategy == ScalingStrategy.MIN_MAX:
        all_mins = df[cols].min().to_numpy().ravel()
        all_maxs = df[cols].max().to_numpy().ravel()
        exprs = ((pl.col(c) - pl.lit(all_mins[i]))/(pl.lit(all_maxs[i] - all_mins[i])) for i,c in enumerate(cols))
        scale_data = [{"min":m, "max":mm} for m, mm in zip(all_mins, all_maxs)]
        scaling_records = ScalingRecord(cols, strategy, scale_data)

    elif strategy == ScalingStrategy.CONST:
        exprs = (pl.col(c)/const for c in cols)
        scale_data = [{"const":const} for _ in cols]
        scaling_records = ScalingRecord(cols, strategy, scale_data)

    else:
        raise ValueError(f"Unknown scaling strategy: {strategy}")

    transformed = df.with_columns(exprs)
    return TransformationResult(transformed=transformed, mapping=scaling_records)

def scale_by(df:pl.DataFrame, rec:ScalingRecord) -> pl.DataFrame:
    
    if rec.strategy == ScalingStrategy.NORMALIZE:
        return df.with_columns(
            (pl.col(f)-pl.lit(v["mean"]))/pl.lit(v["std"]) for f, v in zip(rec.features, rec.values)
        )
    elif rec.strategy == ScalingStrategy.MIN_MAX:
        return df.with_columns(
            (pl.col(f)-pl.lit(v["min"]))/(pl.lit(v["max"] - v["min"])) for f, v in zip(rec.features, rec.values)
        )
    elif rec.strategy == ScalingStrategy.CONST:
        return df.with_columns(
            pl.col(f)/v['const'] for f, v in zip(rec.features, rec.values)
        )    
    else:
        raise ValueError(f"Unknown scaling strategy: {rec.strategy}")

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
    # Here is a rule: Separator must be a single char
    # This is enforced because we want to be able to extract separator from EncoderRecord
    if len(separator) != 1:
        raise ValueError(f"Separator must be a single character for the system to work, not {separator}")

    res = df.to_dummies(columns=one_hot_columns, separator=separator)
    all_mappings = []
    for c in one_hot_columns:
        mapping = {}
        for cc in filter(lambda name: c in name, res.columns):
            # c is original column_name, cc is one-hot created name
            val = cc.replace(c + separator, "") # get original value
            mapping[val] = cc

        all_mappings.append(mapping)

    encoder_rec = EncoderRecord(features=one_hot_columns, strategy=EncodingStrategy.ONE_HOT, mappings=all_mappings)
    return TransformationResult(transformed = res, mapping = encoder_rec)

# def fixed_sized_encode(df:pl.DataFrame, num_cols:list[str], bin_size:int=50) -> TransformationResult:
#     '''Given a continuous variable, take the smallest `bin_size` of them, and call them bin 1, take the next
#     smallest `bin_size` of them and call them bin 2, etc...
    
#     '''
#     pass

# Try to generalize this.
def percentile_encode(df:pl.DataFrame
    , num_cols:list[str]=None
    , exclude:list[str]=None
) -> TransformationResult:
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

    exprs:list[pl.Expr] = []
    all_mappings = []
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
        
        real_mapping = dict(zip(temp_df[c], temp_df["percentile"]))
        # a representation of the mapping.
        repr_mapping = dict(zip(percentile["max"], percentile["percentile"]))
        all_mappings.append(repr_mapping)
        exprs.append(
            pl.col(c).map_dict(real_mapping, default=0).cast(pl.UInt8)
        )
        percentile = percentile.with_columns((
            pl.col("min").cast(pl.Float32),
            pl.col("max").cast(pl.Float32),
            pl.col("cnt").cast(pl.UInt32)
        )) # Need to do this because we need a uniform format in order to stack these columns.

    res = df.with_columns(exprs)
    encoder_rec = EncoderRecord(features=num_list, strategy=EncodingStrategy.PERCENTILE, mappings=all_mappings)
    return TransformationResult(transformed=res, mapping=encoder_rec)

def binary_encode(df:pl.DataFrame
    , binary_cols:list[str]=None
    , exclude:list[str]=None) -> TransformationResult:
    
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

    exprs = []
    mappings = []
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

        first_value = vals[0] if vals[0] is None else str(vals[0])
        mappings.append({first_value: 0, vals[1]: 1, "orig_dtype":dtype_mapping(vals[1])})
        
        exprs.append(
            pl.when(pl.col(b).is_null()).then(0).otherwise(
                pl.when(pl.col(b) < vals[1]).then(0).otherwise(1)
            ).cast(pl.UInt8).alias(b) 
        )

    res = df.with_columns(exprs)
    encoder_rec = EncoderRecord(features=binary_list, strategy=EncodingStrategy.BINARY, mappings=mappings)
    return TransformationResult(transformed = res, mapping = encoder_rec)

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
        ).select(("feature", "value", "mapped_to"))
        mapping_tables.append(table)

    return pl.concat(mapping_tables)

def ordinal_auto_encode(df:pl.DataFrame
    , ordinal_cols:list[str]=None
    , default:int|None=None
    , exclude:list[str]=None) -> TransformationResult:
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
    if isinstance(ordinal_cols, list):
        ordinal_list.extend(ordinal_cols)
    else:
        ordinal_list.extend(get_string_cols(df, exclude=exclude))

    exprs:list[pl.Expr] = []
    all_mappings = []
    for c in ordinal_list:
        sorted_uniques = df.get_column(c).unique().sort()
        mapping:dict[str, int] = dict(zip(sorted_uniques, range(len(sorted_uniques))))
        all_mappings.append(mapping)
        exprs.append(pl.col(c).map_dict(mapping, default=default).cast(pl.UInt32))

    res = df.with_columns(exprs)
    encoder_rec = EncoderRecord(features=ordinal_list, strategy=EncodingStrategy.ORDINAL_AUTO, mappings=all_mappings)
    return TransformationResult(transformed=res, mapping=encoder_rec)

def ordinal_encode(df:pl.DataFrame
    , ordinal_mapping:dict[str, dict[str,int]]
    , default:int|None=None) -> TransformationResult:
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
    
    exprs:list[pl.Expr] = []
    f:list[str] = []
    all_mappings:list[dict[Any, Any]] = []
    for c in ordinal_mapping:
        if c in df.columns:
            mapping = ordinal_mapping[c]
            all_mappings.append(mapping)
            exprs.append(pl.col(c).map_dict(mapping, default=default).cast(pl.UInt32))
        else:
            print(f"Found that column {c} is not in df. Skipped.")

    res = df.with_columns(exprs)
    encoder_rec = EncoderRecord(features=f, strategy=EncodingStrategy.ORDINAL, mappings=all_mappings)
    return TransformationResult(transformed=res, mapping=encoder_rec)

def smooth_target_encode(df:pl.DataFrame, target:str
    , str_cols:list[str]
    , min_samples_leaf:int
    , smoothing:float
    , check_binary:bool=False) -> TransformationResult:
    
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
    
    # Only works for binary target for now 

    # Check if it is binary or not.
    if check_binary:
        target_uniques = list(df.get_column(target).unique().sort())
        if target_uniques != [0,1]:
            raise ValueError(f"The target column {target} must be a binary target with 0 and 1 representing the two classes.")

    p = df.get_column(target).mean() # probability of target = 1
    all_mappings:list[dict[Any, Any]] = []
    exprs:list[pl.Expr] = []
    for c in str_cols:
        ref = df.groupby(c).agg((
            pl.col(target).sum().alias("cnt"),
            pl.col(target).mean().alias("cond_p")
        )).with_columns(
            (1 / (1 + ((-(pl.col("cnt") - pl.lit(min_samples_leaf)))/pl.lit(smoothing)).exp())).alias("alpha")
        ).with_columns(
            (pl.col("alpha") * pl.col("cond_p") + (pl.lit(1) - pl.col("alpha")) * pl.lit(p)).alias("encoded_as")
        ).select(
            pl.col(c).alias("originally_as"),
            pl.col("encoded_as")
        )

        mapping = dict(zip(ref["originally_as"], ref["encoded_as"]))
        all_mappings.append(mapping)
        exprs.append(pl.col(c).map_dict(mapping))
        
    res = df.with_columns(exprs)
    encoder_rec = EncoderRecord(features=str_cols, strategy=EncodingStrategy.TARGET, mappings=all_mappings)
    return TransformationResult(transformed=res, mapping=encoder_rec)

def encode_by(df:pl.DataFrame, rec:EncoderRecord) -> pl.DataFrame:

    # Special cases first
    if rec.strategy == EncodingStrategy.PERCENTILE:
        pass 
    elif rec.strategy == EncodingStrategy.ONE_HOT:
        one_hot_cols = rec.features
        one_hot_map = rec.mappings[0]
        key:str = list(one_hot_map.keys())[0]
        value:str = one_hot_map[key] # must be a string
        separator = value[value.rfind(key) - 1]
        return df.to_dummies(columns=one_hot_cols, separator=separator)

    # Normal case 
    return df.with_columns(
        pl.col(f).map_dict(d) for f,d in zip(rec.features, rec.mappings)
    )