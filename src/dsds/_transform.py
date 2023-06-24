from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, Iterable
from enum import Enum
import polars as pl
import orjson
from .type_alias import PolarsFrame

# Just an archive of some old code which will not be necesssary.

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

    def __init__(self, features:list[str], strategy:ImputationStartegy|str, values:list[float]|np.ndarray):
        self.features = features
        self.strategy = ImputationStartegy(strategy)
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

    def __init__(self, features:list[str], strategy:ScalingStrategy|str, values:list[dict[str, float]]):
        self.features = features
        self.strategy = ScalingStrategy(strategy)
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
        self.strategy = EncodingStrategy(strategy)
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

    def __init__(self, transformed:PolarsFrame, mapping: FitRecord):
        self.transformed = transformed
        self.mapping = mapping
        
    def __iter__(self) -> Iterable[Tuple[PolarsFrame, FitRecord]]:
        return iter((self.transformed, self.mapping))
    
    def materialize(self) -> pl.DataFrame | str:
        return self.mapping.materialize()