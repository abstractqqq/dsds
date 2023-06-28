from pathlib import Path
import polars as pl
from polars import LazyFrame
from dataclasses import dataclass
import pickle
from typing import Iterable, Any, Optional
from polars.type_aliases import IntoExpr
from .type_alias import (
    PolarsFrame
    , StepName
)


@dataclass
class MapDict:
    left_col: str # Join on this column, and this column will be replaced by right and dropped.
    ref: dict # The right table as a dictionary
    right_col: str
    default: Optional[Any]

@dataclass
class Step:
    name:StepName
    associated_data: Iterable[IntoExpr] | MapDict | list[str]
    # First is a with_column, second is a string encoder, third is a drop / a selector


@pl.api.register_lazyframe_namespace("blueprint")
class Blueprint:
    def __init__(self, ldf: LazyFrame):
        self._ldf = ldf
        self.steps:list[Step] = []

    @staticmethod
    def _map_dict(df:PolarsFrame, map_dict:MapDict) -> PolarsFrame:
        temp = pl.from_dict(map_dict.ref) # Always an eager read
        if isinstance(df, pl.LazyFrame): 
            temp = temp.lazy()
        
        if map_dict.default is None:
            return df.join(temp, on = map_dict.left_col).with_columns(
                pl.col(map_dict.right_col).alias(map_dict.left_col)
            ).drop(map_dict.right_col)
        else:
            return df.join(temp, on = map_dict.left_col, how = "left").with_columns(
                pl.col(map_dict.right_col).fill_null(map_dict.default).alias(map_dict.left_col)
            ).drop(map_dict.right_col)

    # Feature Transformations that requires a 1-1 mapping as given by the ref dict. This will be
    # carried out using a join logic to avoid the use of Python UDF.
    def map_dict(self, left_col:str, ref:dict, right_col:str, default:Optional[Any]) -> LazyFrame:
        map_dict = MapDict(left_col = left_col, ref = ref, right_col = right_col, default = default)
        self.steps.append(
            Step(name = "map_dict", associated_data = map_dict)
        )
        output = self._map_dict(self._ldf, map_dict)
        output.blueprint.steps = self.steps # Change "ownership" of this list[Steps] to output.blueprint
        self.steps = [] # Give up self.steps's ownership of the list[Steps] by setting it to an empty list.
        return output

    # Transformations are just with_columns(exprs)
    def with_columns(self, exprs: Iterable[IntoExpr]) -> LazyFrame:
        self.steps.append(
            Step(name = "with_column", associated_data = exprs)
        )
        output = self._ldf.with_columns(exprs)
        output.blueprint.steps = self.steps # Change "ownership" of this list[Steps] to output.blueprint
        self.steps = [] # Give up self.steps's ownership of the list[Steps] by setting it to an empty list.
        return output
    
    # Transformations are just select, used mostly in selector functions
    def select(self, to_select: Iterable[IntoExpr]) -> LazyFrame:
        self.steps.append(
            Step(name = "select", associated_data = to_select)
        )
        output = self._ldf.select(to_select)
        output.blueprint.steps = self.steps # Change "ownership" of this list[Steps] to output.blueprint
        self.steps = [] # Give up self.steps's ownership of the list[Steps] by setting it to an empty list.
        return output
    
    # Transformations that drops, used mostly in removal functions
    def drop(self, drop_cols:Iterable[IntoExpr]) -> LazyFrame:
        self.steps.append(
            Step(name = "drop", associated_data = drop_cols)
        )
        output = self._ldf.drop(drop_cols)
        output.blueprint.steps = self.steps # Change "ownership" of this list[Steps] to output.blueprint
        self.steps = []  # Give up self.steps's ownership of the list[Steps] by setting it to an empty list.
        return output
        
    def preserve(self, path:str|Path):
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    def apply(self, df:PolarsFrame) -> PolarsFrame:
        for s in self.steps:
            if s.name == "drop":
                df = df.drop(s.associated_data)
            elif s.name == "with_column":
                df = df.with_columns(s.associated_data)
            elif s.name == "map_dict":
                df = self._map_dict(df, s.associated_data)
            elif s.name == "select":
                df = df.select(s.associated_data)
            
        return df
