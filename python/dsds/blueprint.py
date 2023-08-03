from pathlib import Path
from polars import LazyFrame, DataFrame
from dataclasses import dataclass, fields
from typing import (
    Any
    , Union
    , Optional
)

from .type_alias import (
    PolarsFrame
    , ActionType
    , PipeFunction
    , ClassifModel
    , RegressionModel
)
import pickle
import polars as pl
import importlib
import logging

logger = logging.getLogger(__name__)

# P = ParamSpec("P")

@dataclass
class MapDict:
    left_col: str # Join on this column, and this column will be replaced by right and dropped.
    ref: dict # The right table as a dictionary
    right_col: str
    default: Optional[Any]

# action + the only non-None field name is a unique identifier for this Step (fully classifies all steps)
@dataclass
class Step:
    action:ActionType
    with_columns: Optional[list[pl.Expr]] = None
    map_dict: Optional[MapDict] = None
    add_func: Optional[dict[str, Any]] = None
    filter: Optional[pl.Expr] = None 
    select_or_drop: Optional[list[str]] = None
    model_step: Optional[dict[str, Any]] = None

    def validate(self) -> bool:
        not_nones:list[str] = []
        for field in fields(self):
            if field.name in ["with_columns", "map_dict", "select_or_drop", "add_func", "filter", "model_step"]:
                if getattr(self, field.name) is not None:
                    not_nones.append(field.name)
            elif field.name == "action":
                continue
            else:
                logger.warning(f"Found unknown action: {field.name}")
                return False

        if len(not_nones) == 1:
            return True
        elif len(not_nones) == 0:
            logger.warning(f"The Step {self.action} does not have any action value associated with it.")
            return False
        else:
            logger.warning(f"The step {self.action} has two action values associated with it: {not_nones}")
            return False
        
    def get_action_value(self) -> Any:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name != "action" and value is not None:
                return value

# Break associated_data into parts?

@pl.api.register_lazyframe_namespace("blueprint")
class Blueprint:
    def __init__(self, ldf: LazyFrame):
        self._ldf = ldf
        self.steps:list[Step] = []

    def as_str(self, n:int) -> str:
        output = ""
        start = max(len(self.steps) + n, 0) if n < 0 else 0
        till = len(self.steps) if n < 0 else min(n, len(self.steps))
        for k,s in enumerate(self.steps):
            if k < start:
                continue
            output += f"Step {k} | Action: {s.action}\n"
            if s.action == "with_columns":
                output += "Details: \n"
                for i,expr in enumerate(s.with_columns):
                    output += f"({i+1}) {expr}\n"
            elif s.action == "add_func":
                d:dict = s.add_func
                output += f"Function Module: {d['module']}, Function Name: {d['name']}\n"
                output += "Parameters:\n"
                for key,value in d["kwargs"].items():
                    output += f"{key} = {value},\n"
            elif s.action == "filter":
                output += f"By condition: {s.filter}\n"
            elif s.action in ("classif", "regression"):
                output += f"Model: {s.model_step['model'].__class__}\n"
                features = s.model_step.get('features', None)
                if features is None:
                    output += "Using all non-target columns as features.\n"
                else:
                    output += f"Using the features {features}\n"
                output += f"Appends {s.model_step['score_col']} to dataframe."
            elif s.action == "map_dict":
                output += f"Encoder/Mapper for column: {s.map_dict.left_col}\n"
                ref_table = pl.from_dict(s.map_dict.ref)
                output += f"The encoding/map is:\n{ref_table.head(10)}\n"                
            else:
                output += str(s.get_action_value())
            output += "\n\n"
            if k > till:
                break
        return output
    
    def show(self, n:int) -> None:
        print(self.as_str(n))

    def __str__(self) -> str:
        return self.as_str(len(self.steps))
    
    def __len__(self) -> int:
        return len(self.steps)
    
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
        
    @staticmethod
    def _process_classif(
        df: PolarsFrame
        , model:ClassifModel
        , features: list[str]
        , target: Optional[str] = None
        , score_idx:int = -1 
        , score_col:str = "model_score"
    ) -> PolarsFrame:
        
        if target is not None:
            if target in features:
                features.remove(target)

        data = df.lazy().collect()
        score = pl.DataFrame({
            score_col: model.predict_proba(data.select(features))[:, score_idx]
        })
        output = pl.concat([data, score], how="horizontal")
        if isinstance(df, pl.LazyFrame):
            return output.lazy()
        return output
    
    @staticmethod
    def _process_regression(
        df: PolarsFrame
        , model:RegressionModel
        , features: list[str]
        , target: Optional[str] = None
        , score_col:str = "model_score"
    ) -> DataFrame:
        
        if target is not None:
            if target in features:
                features.remove(target)

        data = df.lazy().collect()
        score = pl.DataFrame({
            score_col: model.predict(data.select(features)).ravel()
        })
        output = pl.concat([data, score], how="horizontal")
        if isinstance(df, pl.LazyFrame):
            return output.lazy()
        return output

    # Feature Transformations that requires a 1-1 mapping as given by the ref dict. This will be
    # carried out using a join logic to avoid the use of Python UDF.
    def map_dict(self, left_col:str, ref:dict, right_col:str, default:Optional[Any]) -> LazyFrame:
        map_dict = MapDict(left_col = left_col, ref = ref, right_col = right_col, default = default)
        output = Blueprint._map_dict(self._ldf, map_dict)
        output.blueprint.steps = self.steps.copy() 
        output.blueprint.steps.append(
            Step(action = "map_dict", map_dict = map_dict)
        )
        return output
    
    # Shallow copy should work
    # Just make sure exprs are not lazy structures like generators
    
    # Transformations are just with_columns(exprs)
    def with_columns(self, exprs:list[pl.Expr]) -> LazyFrame:
        output = self._ldf.with_columns(exprs)
        output.blueprint.steps = self.steps.copy() # Shallow copy should work
        output.blueprint.steps.append(
            Step(action = "with_columns", with_columns = exprs)
        )
        return output
    
    def filter(self, expr:pl.Expr) -> LazyFrame:
        output = self._ldf.filter(expr)
        output.blueprint.steps = self.steps.copy() # Shallow copy should work
        output.blueprint.steps.append(
            Step(action = "filter", filter = expr)
        )
        return output
    
    # Transformations are just select, used mostly in selector functions
    def select(self, select_cols:list[str]) -> LazyFrame:
        output = self._ldf.select(select_cols)
        output.blueprint.steps = self.steps.copy() 
        output.blueprint.steps.append(
            Step(action = "select", select_or_drop = select_cols)
        )
        return output
    
    # Transformations that drops, used mostly in removal functions
    def drop(self, drop_cols:list[str]) -> LazyFrame:
        output = self._ldf.drop(drop_cols)
        output.blueprint.steps = self.steps.copy() 
        output.blueprint.steps.append(
            Step(action = "drop", select_or_drop = drop_cols)
        )
        return output
    
    def add_func(self
        , df:LazyFrame # The input to the function that needs to be persisted.
        , func:PipeFunction 
        , kwargs:dict[str, Any]
    ) -> LazyFrame:
        # df: The input lazyframe to the function that needs to be persisted. We need this because:
        # When running the function, the reference to df might be changed, therefore losing the steps

        # When this is called, the actual function should be already applied.
        output = self._ldf # .lazy()
        output.blueprint.steps = df.blueprint.steps.copy() 
        output.blueprint.steps.append(
            Step(action="add_func", add_func={"module":func.__module__, "name":func.__name__, "kwargs":kwargs})
        )
        return output

    def add_classif(self
        , model:ClassifModel
        , features: list[str]
        , target: Optional[str] = None
        , score_idx:int = -1 
        , score_col:str = "model_score"
    ) -> LazyFrame:
        '''
        Appends a classification model at given index. This step will collect the lazy frame. All non-target
        column will be used as features.

        Parameters
        ----------
        at
            Index at which to insert the model step
        model
            The trained classification model
        target
            The target of the model, which will not be used in making the prediction. It is only used so that we can 
            remove it from feature list.
        features
            The features the model takes. If none, will use all non-target features.
        score_idx
            The index of the score column in predict_proba you want to append to the dataframe. E.g. -1 will take the 
            score of the positive class in a binary classification
        score_col
            The name of the score column
        '''
        output = Blueprint._process_classif(self._ldf, model, features, target, score_idx, score_col)
        output.blueprint.steps = self.steps.copy()
        output.blueprint.steps.append(
            Step(action = "classif", model_step = {"model":model,
                                                    "target": target,
                                                    "features": features,
                                                    "score_idx": score_idx,
                                                    "score_col":score_col})
        )
        return output
    
    def add_regression(self
        , model:RegressionModel
        , features: list[str]
        , target: Optional[str] = None
        , score_col:str = "model_score"
    ) -> LazyFrame:
        '''
        Appends a classification model at given index. This step will collect the lazy frame. All non-target
        column will be used as features.

        Parameters
        ----------
        at
            Index at which to insert the model step
        model
            The trained classification model
        target
            The target of the model, which will not be used in making the prediction. It is only used so that we can 
            remove it from feature list.
        features
            The features the model takes. If none, will use all non-target features.
        score_idx
            The index of the score column in predict_proba you want to append to the dataframe. E.g. -1 will take the 
            score of the positive class in a binary classification
        score_col
            The name of the score column
        '''        
        output = Blueprint._process_regression(self._ldf, model, features, target, score_col)
        output.blueprint.steps = self.steps.copy()
        output.blueprint.steps.append(
            Step(action = "regression", model_step = {"model":model,
                                                        "target": target,
                                                        "features": features,
                                                        "score_col":score_col})
        )
        return output
    
    def preserve(self, path:Union[str,Path]) -> None:
        '''
        Writes the blueprint to disk as a Python pickle file at the given path.

        Parameters
        ----------
        path
            A valid path to write to
        '''
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def apply(self, df:PolarsFrame, up_to:int=-1, collect:bool=False) -> PolarsFrame:
        '''
        Apply all the steps to the given df. The result will be lazy if df is lazy, and eager if df is eager.

        Parameters
        ----------
        df
            Either an eager or lazy Polars Dataframe
        up_to
            If > 0, will perform the steps up to this number
        collect
            If input is lazy and collect = True, then this will collect the result at the end. If streaming
            collect is desired, please set this to False and collect manually.
        '''
        _up_to = len(self.steps) if up_to <=0 else min(up_to, len(self.steps))
        for i,s in enumerate(self.steps):
            if i < _up_to:
                if s.action == "drop":
                    df = df.drop(s.select_or_drop)
                elif s.action == "with_columns":
                    df = df.with_columns(s.with_columns)
                elif s.action == "map_dict":
                    df = self._map_dict(df, s.map_dict)
                elif s.action == "select":
                    df = df.select(s.select_or_drop)
                elif s.action == "filter":
                    df = df.filter(s.filter)
                elif s.action == "add_func":
                    module, name = s.add_func["module"], s.add_func["name"]
                    func = getattr(importlib.import_module(module), name)
                    df = df.pipe(func, **s.add_func["kwargs"])
                elif s.action == "classif":
                    df = df.pipe(Blueprint._process_classif, **s.model_step)
                elif s.action == "regression":
                    df = df.pipe(Blueprint._process_regression, **s.model_step)
            else:
                break

        if isinstance(df, pl.LazyFrame) and collect:
            return df.collect()
        return df

def from_pkl(path: Union[str,Path]) -> Blueprint:
    with open(path, "rb") as f:
        obj = pickle.loads(f.read())
        if isinstance(obj, Blueprint):
            return obj
        else:
            raise ValueError("The object in the pickled file is not a Blueprint object.")


