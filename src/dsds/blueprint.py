import pickle
import polars as pl
# import importlib
from pathlib import Path
from polars import LazyFrame
from dataclasses import dataclass
from typing import (
    Any
    , Iterable
    , Optional
)
from polars.type_aliases import IntoExpr
from .type_alias import (
    PolarsFrame
    , ActionType
    # , PipeFunction
)


@dataclass
class MapDict:
    left_col: str # Join on this column, and this column will be replaced by right and dropped.
    ref: dict # The right table as a dictionary
    right_col: str
    default: Optional[Any]

@dataclass
class Step:
    action:ActionType
    associated_data: Iterable[IntoExpr] | MapDict | list[str]
    # First is a with_column, second is a string encoder, third is a drop/select/apply_func


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
            Step(action = "map_dict", associated_data = map_dict)
        )
        output = self._map_dict(self._ldf, map_dict)
        output.blueprint.steps = self.steps # Change "ownership" of this list[Steps] to output.blueprint
        self.steps = [] # Give up self.steps's ownership of the list[Steps] by setting it to an empty list.
        return output

    # Transformations are just with_columns(exprs)
    def with_columns(self, exprs:Iterable[IntoExpr]) -> LazyFrame:
        self.steps.append(
            Step(action = "with_column", associated_data = list(exprs))
        )
        output = self._ldf.with_columns(exprs)
        output.blueprint.steps = self.steps # Change "ownership" of this list[Steps] to output.blueprint
        self.steps = [] # Give up self.steps's ownership of the list[Steps] by setting it to an empty list.
        return output
    
    # Transformations are just select, used mostly in selector functions
    def select(self, to_select:list[str]) -> LazyFrame:
        self.steps.append(
            Step(action = "select", associated_data = to_select)
        )
        output = self._ldf.select(to_select)
        output.blueprint.steps = self.steps # Change "ownership" of this list[Steps] to output.blueprint
        self.steps = [] # Give up self.steps's ownership of the list[Steps] by setting it to an empty list.
        return output
    
    # Transformations that drops, used mostly in removal functions
    def drop(self, drop_cols:list[str]) -> LazyFrame:
        self.steps.append(
            Step(action = "drop", associated_data = drop_cols)
        )
        output = self._ldf.drop(drop_cols)
        output.blueprint.steps = self.steps # Change "ownership" of this list[Steps] to output.blueprint
        self.steps = []  # Give up self.steps's ownership of the list[Steps] by setting it to an empty list.
        return output
    
    # # Functional steps are steps like upsample/downsample, which can be persisted in pipeline, but 
    # # may not be repeatable.
    # def add_functional_step(self, func:PipeFunction):
    #     self.steps.append(
    #         Step(action="apply_func", associated_data=[func.__module__, func.__name__])
    #     )
        
    def preserve(self, path:str|Path):
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    def apply(self, df:PolarsFrame) -> PolarsFrame:
        for s in self.steps:
            if s.action == "drop":
                df = df.drop(s.associated_data)
            elif s.action == "with_column":
                df = df.with_columns(s.associated_data)
            elif s.action == "map_dict":
                df = self._map_dict(df, s.associated_data)
            elif s.action == "select":
                df = df.select(s.associated_data)
            
        return df
