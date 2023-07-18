import pickle
import polars as pl
import importlib
import polars.selectors as cs
from pathlib import Path
from polars import LazyFrame
from dataclasses import dataclass
from typing import (
    Any
    , Iterable
    , Optional
    , Callable
    , Concatenate
    , ParamSpec
)
from polars.type_aliases import IntoExpr
from .type_alias import (
    PolarsFrame
    , ActionType
    # , PipeFunction
)

P = ParamSpec("P")

@dataclass
class MapDict:
    left_col: str # Join on this column, and this column will be replaced by right and dropped.
    ref: dict # The right table as a dictionary
    right_col: str
    default: Optional[Any]

@dataclass
class Step:
    action:ActionType
    associated_data: Iterable[IntoExpr] | MapDict | list[str] | cs._selector_proxy_ | dict[str, Any] | pl.Expr
    # First is everything that can be done with with_columns (Iterable[IntoExpr], but list[pl.Expr] is recommended)
    # Second is a 1-to-1 encoder (MapDict)
    # Third is a drop/select (list[str] and cs._selector_proxy_)
    # Fourth is add_func (dict[str, Any])
    # Fifth is a filter statement (pl.Expr)


@pl.api.register_lazyframe_namespace("blueprint")
class Blueprint:
    def __init__(self, ldf: LazyFrame):
        self._ldf = ldf
        self.steps:list[Step] = []
        self.target:str = ""
        self.model = None

    def __str__(self) -> str:
        output = ""
        for k,s in enumerate(self.steps):
            output += f"Step {k} | Action: {s.action}\n"
            if s.action == "with_columns":
                output += "Details: \n"
                for i,expr in enumerate(s.associated_data):
                    output += f"({i+1}) {expr}\n"
            elif s.action == "apply_func":
                d:dict = s.associated_data
                output += f"Function Module: {d['module']}, Function Name: {d['name']}\n"
                output += "Parameters:\n"
                for k,v in d["kwargs"].items():
                    output += f"{k} = {v},\n"
            elif s.action == "filter":
                output += f"By condition: {s.associated_data}\n"
            else:
                output += str(s.associated_data)

            output += "\n\n"
        return output
    
    def _ipython_display_(self):
        print(self)

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
        output = self._map_dict(self._ldf, map_dict)
        output.blueprint.steps = self.steps.copy() 
        output.blueprint.steps.append(
            Step(action = "map_dict", associated_data = map_dict)
        )
        return output
    
    # Shallow copy should work
    # Just make sure exprs are not lazy structures like generators
    
    # Transformations are just with_columns(exprs)
    def with_columns(self, exprs:Iterable[IntoExpr]) -> LazyFrame:
        output = self._ldf.with_columns(exprs)
        output.blueprint.steps = self.steps.copy() # Shallow copy should work
        output.blueprint.steps.append(
            Step(action = "with_columns", associated_data = exprs)
        )
        return output
    
    def filter(self, expr:pl.Expr) -> LazyFrame:
        output = self._ldf.filter(expr)
        output.blueprint.steps = self.steps.copy() # Shallow copy should work
        output.blueprint.steps.append(
            Step(action = "filter", associated_data = expr)
        )
        return output
    
    # Transformations are just select, used mostly in selector functions
    def select(self, to_select:list[str]) -> LazyFrame:
        output = self._ldf.select(to_select)
        output.blueprint.steps = self.steps.copy() 
        output.blueprint.steps.append(
            Step(action = "select", associated_data = to_select)
        )
        return output
    
    # Transformations that drops, used mostly in removal functions
    def drop(self, drop_cols:list[str]) -> LazyFrame:
        output = self._ldf.drop(drop_cols)
        output.blueprint.steps = self.steps.copy() 
        output.blueprint.steps.append(
            Step(action = "drop", associated_data = drop_cols)
        )
        return output
    
    # This doesn't work with the pipeline right now because it clears all steps before this.
    def add_func(self
        , df:LazyFrame # The input to the function that needs to be persisted.
        , func:Callable[Concatenate[LazyFrame, P], LazyFrame]
        , kwargs:dict[str, Any]
    ) -> LazyFrame:
        # df: The input lazyframe to the function that needs to be persisted. We need this because:
        # When running the function, the reference to df might be changed, therefore losing the steps

        # When this is called, the actual function should be already applied.
        output = self._ldf # .lazy()
        output.blueprint.steps = df.blueprint.steps.copy() 
        output.blueprint.steps.append(
            Step(action="add_func", associated_data={"module":func.__module__, "name":func.__name__, "kwargs":kwargs})
        )
        return output
        
    def preserve(self, path:str|Path):
        '''
        Writes the blueprint to disk as a Python pickle file at the given path.
        '''
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    def apply(self, df:PolarsFrame, up_to:int=-1) -> PolarsFrame:
        '''
        Apply all the steps to the given df. The result will be lazy if df is lazy, and eager if df is eager.

        Parameters
        ----------
        df
            Either an eager or lazy Polars Dataframe
        up_to
            If > 0, will perform the steps up to this number
        '''
        _up_to = len(self.steps) if up_to <=0 else min(up_to, len(self.steps))
        for i,s in enumerate(self.steps):
            if i < _up_to:
                if s.action == "drop":
                    df = df.drop(s.associated_data)
                elif s.action == "with_columns":
                    df = df.with_columns(s.associated_data)
                elif s.action == "map_dict":
                    df = self._map_dict(df, s.associated_data)
                elif s.action == "select":
                    df = df.select(s.associated_data)
                elif s.action == "filter":
                    df = df.filter(s.associated_data)
                elif s.action == "add_func":
                    func = getattr(importlib.import_module(s.associated_data["module"]), s.associated_data["name"])
                    df = df.pipe(func, **s.associated_data["kwargs"])
            else:
                break
        return df
