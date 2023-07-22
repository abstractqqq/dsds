import pickle
import polars as pl
import importlib
import polars.selectors as cs
from pathlib import Path
from polars import LazyFrame, DataFrame
from dataclasses import dataclass
from typing import (
    Any
    , Union
    , Iterable
    , Optional
    # , Callable
    # , Concatenate
    # , ParamSpec
)
from polars.type_aliases import IntoExpr
from .type_alias import (
    PolarsFrame
    , ActionType
    , PipeFunction
    , ClassifModel
    # , RegressionModel
)

# P = ParamSpec("P")

@dataclass
class MapDict:
    left_col: str # Join on this column, and this column will be replaced by right and dropped.
    ref: dict # The right table as a dictionary
    right_col: str
    default: Optional[Any]

@dataclass
class Step:
    action:ActionType
    associated_data: Union[Iterable[IntoExpr], MapDict, list[str], cs._selector_proxy_, dict[str, Any], pl.Expr]
    # First is everything that can be done with with_columns (Iterable[IntoExpr], but list[pl.Expr] is recommended)
    # Second is a 1-to-1 encoder (MapDict)
    # Third is a drop/select (list[str] and cs._selector_proxy_)
    # Fourth is add_func (dict[str, Any]), or add_classif
    # Fifth is a filter statement (pl.Expr)


@pl.api.register_lazyframe_namespace("blueprint")
class Blueprint:
    def __init__(self, ldf: LazyFrame):
        self._ldf = ldf
        self.steps:list[Step] = []
        # self.model = None
        # self.source = ldf.clone()

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
            elif s.action == "classif":
                output += f"Classification Model: {s.associated_data['model'].__class__}\n"
                features = s.associated_data.get('features', None)
                if features is None:
                    output += "Using all non-target columns as features.\n"
                else:
                    output += f"Using the features {s.associated_data['features']}\n"
                output += f"Appends {s.associated_data['score_col']} to dataframe."
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
        
    @staticmethod
    def _process_classif(
        df: PolarsFrame
        , model:ClassifModel
        , target: Optional[str] = None
        , features: Optional[list[str]] = None
        , score_idx:int = -1 
        , score_col:str = "model_score"
    ) -> DataFrame:
        
        if features is None:
            features = df.columns
        if target is not None:
            if target in features:
                features.remove(target)

        data = df.lazy().collect()
        return data.insert_at_idx(0, pl.Series(
            score_col, model.predict_proba(data.select(features))[:, score_idx]
        ))


    # Feature Transformations that requires a 1-1 mapping as given by the ref dict. This will be
    # carried out using a join logic to avoid the use of Python UDF.
    def map_dict(self, left_col:str, ref:dict, right_col:str, default:Optional[Any]) -> LazyFrame:
        map_dict = MapDict(left_col = left_col, ref = ref, right_col = right_col, default = default)
        output = Blueprint._map_dict(self._ldf, map_dict)
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

    def add_classif(self
        , model:ClassifModel
        , target: Optional[str] = None
        , features: Optional[list[str]] = None
        , score_idx:int = -1 
        , score_col:str = "model_score"
    ) -> LazyFrame:
        '''
        Appends a classification model to the end of the pipeline. This step will collect the lazy frame. All non-target
        column will be used as features.

        Parameters
        ----------
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
        output = self._process_classif(self._ldf, model, target, features, score_idx, score_col).lazy()
        output.blueprint.steps = self.steps.copy()
        output.blueprint.steps.append(
            Step(action = "classif", associated_data={"model":model,
                                                      "target": target,
                                                      "features": features,
                                                      "score_idx": score_idx,
                                                      "score_col":score_col})
        )
        return output

    def insert_classif(self
        , at: int
        , model:ClassifModel
        , target: Optional[str] = None
        , features: Optional[list[str]] = None
        , score_idx:int = -1 
        , score_col:str = "model_score"
    ) -> LazyFrame:
        '''
        Inserts a classification model at given index. This step will collect the lazy frame. All non-target
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
        output = self._process_classif(model, target, features, score_idx, score_col).lazy()
        output.blueprint.steps = self.steps.copy()
        output.blueprint.steps.insert(at,
            Step(action = "classif", associated_data={"model":model,
                                                      "target": target,
                                                      "features": features,
                                                      "score_idx": score_idx,
                                                      "score_col":score_col})
        )
        return output   

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
                elif s.action == "classif":
                    if isinstance(df, LazyFrame):
                        df = df.pipe(self._process_classif, **s.associated_data).lazy()
                    else:
                        df = df.pipe(self._process_classif, **s.associated_data)
            else:
                break
        return df
