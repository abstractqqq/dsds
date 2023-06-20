from __future__ import annotations

from .prescreen import (
    get_bool_cols
    , get_numeric_cols
    , get_string_cols
    , get_unique_count
    , dtype_mapping
)

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import orjson
import polars as pl
import numpy as np 
from enum import Enum
from typing import Any, Tuple, Iterable, Optional

# A lot of companies are still using Python < 3.10
# So I am not using match statements
# Well, it does say in project description that we need Python 3.10.

logger = logging.getLogger(__name__)

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
    ORDINAL_AUTO = "ORDINAL-AUTO"
    TARGET = "TARGET"
    ONE_HOT = "ONE-HOT"
    BINARY = "BINARY"
    PERCENTILE = "PERCENTILE"

# It is highly recommended that this should be a dataclass and serializable by orjson.
class FitRecord(ABC):

    @abstractmethod
    def materialize(self) -> pl.DataFrame | str:
        # A pretty way to print or visualize itself, 
        # or organize self to something more useful than a data structure.
        pass 

    @abstractmethod
    def transform(self, df:pl.DataFrame) -> pl.DataFrame:
        # Transform according to the record.
        pass

@dataclass
class ImputationRecord(FitRecord):
    features:list[str]
    strategy:ImputationStartegy
    values:list[float]|np.ndarray

    def __init__(self, features:list[str], strategy:EncodingStrategy|str, values:list[float]|np.ndarray):
        self.features = features
        self.strategy = ImputationStartegy(strategy) if isinstance(strategy, str) else strategy
        self.values = values

    def __iter__(self) -> Iterable:
        return zip(self.features, [self.strategy]*len(self.features), self.values)
    
    def __str__(self) -> str:
        return orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY).decode()
    
    def materialize(self) -> pl.DataFrame:
        return pl.from_records(list(self), schema=["feature", "imputation_strategy", "value_used"])
    
    def transform(self, df:pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(f).fill_null(v) for f, v in zip(self.features, self.values)
        )
    
@dataclass
class ScalingRecord(FitRecord):
    features:list[str]
    strategy:ScalingStrategy
    values:list[dict[str, float]]

    def __init__(self, features:list[str], strategy:EncodingStrategy|str, values:list[dict[str, float]]):
        self.features = features
        self.strategy = ScalingStrategy(strategy) if isinstance(strategy, str) else strategy
        self.values = values

    def __iter__(self) -> Iterable:
        return zip(self.features, [self.strategy]*len(self.features), self.values)
    
    def __str__(self) -> str:
        return orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY).decode()
    
    def materialize(self) -> pl.DataFrame:
        vals = (orjson.dumps(v, option=orjson.OPT_SERIALIZE_NUMPY).decode() for v in self.values)
        presentable =  zip(self.features, [self.strategy]*len(self.features), vals)
        return pl.from_records(list(presentable), schema=["feature", "scaling_strategy", "scaling_meta_data"])
    
    def transform(self, df:pl.DataFrame) -> pl.DataFrame:

        if self.strategy == ScalingStrategy.NORMALIZE:
            return df.with_columns(
                (pl.col(f)-pl.lit(v["mean"]))/pl.lit(v["std"]) for f, v in zip(self.features, self.values)
            )
        elif self.strategy == ScalingStrategy.MIN_MAX:
            return df.with_columns(
                (pl.col(f)-pl.lit(v["min"]))/(pl.lit(v["max"] - v["min"])) for f, v in zip(self.features, self.values)
            )
        elif self.strategy == ScalingStrategy.CONST:
            return df.with_columns(
                pl.col(f)/v['const'] for f, v in zip(self.features, self.values)
            )    
        else:
            raise ValueError(f"Unknown scaling strategy: {self.strategy}")

@dataclass
class EncoderRecord(FitRecord):
    features:list[str]
    strategy:EncodingStrategy
    mappings:list[dict]

    ### FOR str encoders, mapping looks like "dict[str, float]", except one-hot. See one-hot for more info.
    ### For numeric encoder, like percentile encoder, the key of the mapping is of type str despite the fact that
    ### it is a number. This is because json has to have str as keys. See percentile_encode for more info.

    def __init__(self, features:list[str], strategy:EncodingStrategy|str, mappings:list[dict[Any, Any]]):
        self.features = features
        self.strategy = EncodingStrategy(strategy) if isinstance(strategy, str) else strategy
        self.mappings = mappings

    def __iter__(self) -> Iterable:
        return zip(self.features, [self.strategy]*len(self.features), self.mappings)
    
    def __str__(self) -> str:
        return orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY|orjson.OPT_NON_STR_KEYS).decode()
    
    def materialize(self) -> pl.DataFrame:
        vals = (orjson.dumps(v, option=orjson.OPT_SERIALIZE_NUMPY|orjson.OPT_NON_STR_KEYS).decode() for v in self.mappings)
        presentable =  zip(self.features, [self.strategy]*len(self.features), vals)
        return pl.from_records(list(presentable), schema=["feature", "encoding_strategy", "maps"])
    
    ###
    # NEED TO FIND WAYS TO OPTIMIZE ENCODINGS FOR Numeric values...
    ###

    @staticmethod
    def _find_first_index_of_smaller(u:float, order:list[Tuple[float, int]]) -> int:
        order.sort(key=lambda x: x[1])
        for v, i in order: # order looks like [(18.21, 1), (22.32, 2), ...]
            if u <= v:
                return i
        # percentile max out at 100. It is possible that in future data, there will be some
        # that is > existing max. So assign all that to 101
        return 101 

    def transform(self, df:pl.DataFrame) -> pl.DataFrame:
        # Special cases first
        if self.strategy == EncodingStrategy.PERCENTILE:
            for i,f in enumerate(self.features):
                # Construct a new series for each column. SLOW SLOW SLOW...

                # If this comes from a blue_print, then we will get a dict with str keys
                # because JSON KEY IS ALWAYS A STR.
                # If we are running this after generating this record, the original key is 
                # numeric. So either way, this works.
                order = [(float(v), p) for v, p in self.mappings[i].items()] 
                percentiles = []
                already_mapped = {}
                for v in df.get_column(f):
                    if v is None or np.isnan(v) or np.isneginf(v): # To 0
                        percentiles.append(0) 
                    else:
                        if v in already_mapped:
                            percentiles.append(already_mapped[v])
                        else:
                            percentile = self._find_first_index_of_smaller(v, order)
                            already_mapped[v] = percentile
                            percentiles.append(percentile)
                
                new_f = pl.Series(f, percentiles).cast(pl.UInt8)
                df.replace_at_idx(df.find_idx_by_name(f), new_f)
                
            return df
        
        elif self.strategy == EncodingStrategy.ONE_HOT:
            one_hot_cols = self.features
            one_hot_map = self.mappings[0] # One hot mapping only has 1 mapping in the list.
            key:str = list(one_hot_map.keys())[0]
            value:str = one_hot_map[key] # must be a string
            separator = value[value.rfind(key) - 1]
            return df.to_dummies(columns=one_hot_cols, separator=separator)

        # Normal case 
        return df.with_columns(
            pl.col(f).map_dict(d) for f,d in zip(self.features, self.mappings)
        )

class FitTransform:

    def __init__(self, transformed:pl.DataFrame, mapping: FitRecord):
        self.transformed = transformed
        self.mapping = mapping
        
    def __iter__(self) -> Iterable[Tuple[pl.DataFrame, FitRecord]]:
        return iter((self.transformed, self.mapping))
    
    def materialize(self) -> pl.DataFrame | str:
        return self.mapping.materialize()


def check_columns_types(df:pl.DataFrame, cols:Optional[list[str]]=None) -> str:
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
    
    if len(types) > 0:
        return "|".join(types)
    else:
        return "unknown"


def impute(df:pl.DataFrame
    , cols:list[str]
    , strategy:ImputationStartegy|str = ImputationStartegy.MEDIAN
    , const:int = 1
) -> FitTransform:
    '''
        Arguments:
            df:
            cols:
            strategy:
            const: only uses this value if strategy = ImputationStartegy.CONST
    
    '''

    s = ImputationStartegy(strategy.replace("-","_")) if isinstance(strategy, str) else strategy
    # Given Strategy, define expressions
    if s == ImputationStartegy.MEDIAN:
        all_medians = df[cols].median().to_numpy().ravel()
        exprs = (pl.col(c).fill_null(all_medians[i]) for i,c in enumerate(cols))
        impute_record = ImputationRecord(cols, strategy, all_medians)

    elif s == ImputationStartegy.MEAN:
        all_means = df[cols].mean().to_numpy().ravel()
        exprs = (pl.col(c).fill_null(all_means[i]) for i,c in enumerate(cols))
        impute_record = ImputationRecord(cols, strategy, all_means)

    elif s == ImputationStartegy.CONST:
        exprs = (pl.col(c).fill_null(const) for c in cols)
        impute_record = ImputationRecord(cols, strategy, np.full(shape=len(cols), fill_value=const))

    elif s == ImputationStartegy.MODE:
        all_modes = df.select(pl.col(c).mode() for c in cols).to_numpy().ravel()
        exprs = (pl.col(c).fill_null(all_modes[i]) for i,c in enumerate(cols))
        impute_record = ImputationRecord(cols, strategy, all_modes)

    else:
        raise ValueError(f"Unknown imputation strategy: {s}")

    transformed = df.with_columns(exprs)
    return FitTransform(transformed=transformed, mapping=impute_record)

def scale(df:pl.DataFrame
    , cols:list[str]
    , strategy:ScalingStrategy=ScalingStrategy.NORMALIZE
    , const:int = 1
) -> FitTransform:
    
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
    
    s = ScalingStrategy(strategy.replace("-","_")) if isinstance(strategy, str) else strategy
    if s == ScalingStrategy.NORMALIZE:
        all_means = df[cols].mean().to_numpy().ravel()
        all_stds = df[cols].std().to_numpy().ravel()
        exprs = (((pl.col(c) - pl.lit(all_means[i]))/(pl.lit(all_stds[i])) for i,c in enumerate(cols)))
        scale_data = [{"mean":m, "std":s} for m,s in zip(all_means, all_stds)]
        scaling_records = ScalingRecord(cols, strategy, scale_data)

    elif s == ScalingStrategy.MIN_MAX:
        all_mins = df[cols].min().to_numpy().ravel()
        all_maxs = df[cols].max().to_numpy().ravel()
        exprs = ((pl.col(c) - pl.lit(all_mins[i]))/(pl.lit(all_maxs[i] - all_mins[i])) for i,c in enumerate(cols))
        scale_data = [{"min":m, "max":mm} for m, mm in zip(all_mins, all_maxs)]
        scaling_records = ScalingRecord(cols, strategy, scale_data)

    elif s == ScalingStrategy.CONST:
        exprs = (pl.col(c)/const for c in cols)
        scale_data = [{"const":const} for _ in cols]
        scaling_records = ScalingRecord(cols, strategy, scale_data)

    else:
        raise ValueError(f"Unknown scaling strategy: {strategy}")

    transformed = df.with_columns(exprs)
    return FitTransform(transformed=transformed, mapping=scaling_records)

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

def one_hot_encode(df:pl.DataFrame, cols:Optional[list[str]]=None, separator:str="_") -> FitTransform:
    '''One hot encoding. The separator must be a single character.'''

    # Here is a rule: Separator must be a single char
    # This is enforced because we want to be able to extract separator from EncoderRecord
    if len(separator) != 1:
        raise ValueError(f"Separator must be a single character for the system to work, not {separator}")
    
    str_cols = []
    if isinstance(cols, list):
        types = check_columns_types(df, cols)
        if types != "string":
            raise ValueError(f"One-hot encoding can only be used on string columns, not {types} types.")
        str_cols.extend(cols)
    else:
        str_cols = get_string_cols(df)

    res = df.to_dummies(columns=str_cols, separator=separator)
    all_mappings = []
    for c in str_cols:
        mapping = {}
        for cc in filter(lambda name: c in name, res.columns):
            # c is original column_name, cc is one-hot created name
            val = cc.replace(c + separator, "") # get original value
            mapping[val] = cc

        all_mappings.append(mapping)

    encoder_rec = EncoderRecord(features=str_cols, strategy=EncodingStrategy.ONE_HOT, mappings=all_mappings)
    return FitTransform(transformed = res, mapping = encoder_rec)

# def fixed_sized_encode(df:pl.DataFrame, num_cols:list[str], bin_size:int=50) -> TransformationResult:
#     '''Given a continuous variable, take the smallest `bin_size` of them, and call them bin 1, take the next
#     smallest `bin_size` of them and call them bin 2, etc...
    
#     '''
#     pass

# Try to generalize this.
def percentile_encode(df:pl.DataFrame
    , cols:list[str]=None
    , exclude:list[str]=None
) -> FitTransform:
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
    if isinstance(cols, list):
        types = check_columns_types(df, cols)
        if types != "numeric":
            raise ValueError(f"Percentile encoding can only be used on numeric columns, not {types} types.")
        num_list.extend(cols)
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
    return FitTransform(transformed=res, mapping=encoder_rec)

def binary_encode(df:pl.DataFrame
    , cols:Optional[list[str]]=None
    , exclude:Optional[list[str]]=None
) -> FitTransform:
    
    '''Encode the given columns as binary values.

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

    exprs = []
    mappings = []
    binary_list = []
    if isinstance(cols, list):
        binary_list.extend(cols)
    else:
        str_cols = get_string_cols(df)
        exclude = [] if exclude is None else exclude
        binary_columns = get_unique_count(df)\
            .filter( # Binary + Not Exclude + String
                (pl.col("n_unique") == 2) & (~pl.col("column").is_in(exclude)) & (pl.col("column").is_in(str_cols))
            ).get_column("column")

        # Binary numericals are kept the way they are.
        binary_list.extend(binary_columns)     
    
    # Doing some repetitive operations here, but I am not sure how I can get all the data in one go.
    for b in binary_list:
        vals = df.get_column(b).unique().to_list()
        logger.info(f"Transforming {b} into a binary column with [0, 1] ...")
        if len(vals) != 2:
            logger.warning(f"Found {b} has {len(vals)} unique values instead of 2. Not a binary variable. Ignored.")
            continue
        if vals[0] is None: # Weird code, but we need this case.
            pass
        elif vals[1] is None:
            vals[0], vals[1] = vals[1], vals[0]
        else:
            vals.sort()

        # In Python, None can be a dictionary key.
        mappings.append({vals[0]: 0, vals[1]: 1})
        exprs.append(
            pl.when(pl.col(b).is_null()).then(0).otherwise(
                pl.when(pl.col(b) < vals[1]).then(0).otherwise(1)
            ).cast(pl.UInt8).alias(b) 
        )

    res = df.with_columns(exprs)
    encoder_rec = EncoderRecord(features=binary_list, strategy=EncodingStrategy.BINARY, mappings=mappings)
    return FitTransform(transformed = res, mapping = encoder_rec)

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
    , cols:list[str]=None
    , default:int|None=None
    , exclude:Optional[list[str]]=None
) -> FitTransform:
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

    exprs:list[pl.Expr] = []
    all_mappings = []
    for c in ordinal_list:
        sorted_uniques = df.get_column(c).unique().sort()
        mapping:dict[str, int] = dict(zip(sorted_uniques, range(len(sorted_uniques))))
        all_mappings.append(mapping)
        exprs.append(pl.col(c).map_dict(mapping, default=default).cast(pl.UInt32))

    res = df.with_columns(exprs)
    encoder_rec = EncoderRecord(features=ordinal_list, strategy=EncodingStrategy.ORDINAL_AUTO, mappings=all_mappings)
    return FitTransform(transformed=res, mapping=encoder_rec)

def ordinal_encode(df:pl.DataFrame
    , ordinal_mapping:dict[str, dict[str,int]]
    , default:int|None=None
) -> FitTransform:
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
            logger.warning(f"Found that column {c} is not in df. Skipped.")

    res = df.with_columns(exprs)
    encoder_rec = EncoderRecord(features=f, strategy=EncodingStrategy.ORDINAL, mappings=all_mappings)
    return FitTransform(transformed=res, mapping=encoder_rec)

def smooth_target_encode(
    df:pl.DataFrame
    , target:str
    , cols:list[str]
    , min_samples_leaf:int
    , smoothing:float
    , check_binary:bool=True
) -> FitTransform:
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
        target_uniques = df.get_column(target).unique()
        if len(target_uniques) != 2 or (not (0 in target_uniques and 1 in target_uniques)):
            raise ValueError(f"The target column {target} must be a binary target with 0s and 1s.")

    p = df.get_column(target).mean() # probability of target = 1
    all_mappings:list[dict[Any, Any]] = []
    exprs:list[pl.Expr] = []
    # If c has null, null will become a group when we group by.
    for c in str_cols:
        ref = df.groupby(c).agg(
            pl.col(target).sum().alias("cnt"),
            pl.col(target).mean().alias("cond_p")
        ).with_columns(
            (1/(1 + ((-(pl.col("cnt") - pl.lit(min_samples_leaf)))/pl.lit(smoothing)).exp())).alias("alpha")
        ).with_columns(
            (pl.col("alpha") * pl.col("cond_p") + (pl.lit(1) - pl.col("alpha")) * pl.lit(p)).alias("encoded_as")
        )
        
        mapping = dict(zip(ref[c], ref["encoded_as"]))
        all_mappings.append(mapping)
        exprs.append(pl.col(c).map_dict(mapping))
        
    res = df.with_columns(exprs)
    encoder_rec = EncoderRecord(features=str_cols, strategy=EncodingStrategy.TARGET, mappings=all_mappings)
    return FitTransform(transformed=res, mapping=encoder_rec)