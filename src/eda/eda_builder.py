from __future__ import annotations

from .eda_prescreen import *
from .eda_selection import *
from .eda_transformations import *
from dataclasses import dataclass
import polars as pl
import pandas as pd
from typing import Self, Union, Optional, Any, Callable
from enum import Enum
from functools import partial
import os, time, json # Replace json by orjson?
from pathlib import Path

################################################################################################
# WORK IN PROGRESS
################################################################################################

class TransformedData:
    pass
    
    # def __init__(self):
    #     self.df:pl.DataFrame = None
    #     self.target:str = ""
    #     self.log:list[str] = []
    #     self.mappings:list[pl.DataFrame] = []


# We need this abstraction for the following reasons
# 1. In the future, I want to add "checkpoint" (save to local)
# But if something is saved to local before, I want the plan to 
# notice it and read directly to local and skip the "checkpointed" steps.
# 2. Better organization for the execution.
#

# This is just a builtin mapping for desc.
class BuiltinExecutions(Enum):
    NULL_REMOVAL = "Remove columns with more than {:.2f}% nulls."
    VAR_REMOVAL = "Remove columns with less than {} variance. (Not recommended.)"
    CONST_REMOVAL = "Remove columns that are constants."
    COL_REMOVAL = "Remove the given columns if they exist in dataframe."
    REGX_REMOVAL = "Remove all columns whose names satisfy the regex rule."
    BINARY_ENCODE = "Encode given columns into binary [0,1] values."
    ORDINAL_ENCODE = "Encode string values of given columns into numbers with given mapping."
    ORDINAL_AUTO_ENCODE = "Encode string values of given columns into numbers with inferred ordering."
    TARGET_ENCODE = "Encode string values using the target encoding algorithm."
    ONE_HOT_ENCODE = "Encode string values of given columns by the one-hot-encoding technique. (No drop first option)."
    ENCODER_RECORD = "Encode by a given encoder record."
    SCALE = "Scale using specified the {} scaling method."
    IMPUTE = "Impute using specified the {} imputation method."
    CHECKPOINT = "Unavailable for now."

@dataclass
class ExecStep():
    name:str
    module_path:str|Path|None
    desc:str
    args:dict[str, Any]
    is_transformation:bool = False
    # Is this a transfromation call? 
    # If so, the output is expected to be of type TransformationResult.
    # If not, the output is expected to be of type pl.DataFrame.
    is_custom:bool = False # If is custom, module path will be recorded.

    def get_args(self) -> str:
        return self.args
    
    # __str__

    # def __init__(self, step:_BuiltinExecutions, desc:str, func:Callable):
    #     self.name = step.name
    #     self.desc = desc
    #     self.func = func

@dataclass
class ExecPlan():
    steps:list[ExecStep] = []

    def add(self, step:ExecStep) -> None:
        self.steps.append(step)

    def add_step(self, func:Callable
                , desc:str, args:dict[str, Any]
                , is_transf:bool=False) -> None:
        self.steps.append(
            ExecStep(func.__name__, None, desc, args, is_transf, True)
        )

    def add_custom_step(self, func:Callable
                , desc:str, args:dict[str, Any]
                , is_transf:bool=False) -> None:
        
        self.steps.append(
            ExecStep(func.__name__, func.__module__, desc, args, is_transf)
        )

class TransformationBuilder:

    def __init__(self, project_name:str="my_project") -> Self:
        self.df:pl.DataFrame = None
        self.target:str = ""
        self.project_name:str = project_name
        self.ready_to_go:bool = False
        self.execution_plan:ExecPlan = ExecPlan()
        return self
    
    def _is_ready(self) -> bool:
        return isinstance(self.df, pl.DataFrame) and not self.df.is_empty() and self.target in self.df.columns and self.target != ""
    
    ### Project meta data section
    def set_target(self, target:str) -> Self: 
        self.target = target
        self.ready_to_go = self._is_ready()
        return self 
    
    def set_project_name(self, pname:str) -> Self:
        self.project_name = pname
        return self

    ### End of project meta data section
    
    ### IO Section 
    def read_csv_from(self, path:str|Path, csv_args:Optional[dict[str, Any]] = None) -> Self:
        args = {} if csv_args is None else csv_args.copy()
        if "source" not in args:
            args["source"] = path

        self.df = pl.read_csv(**args)
        self.ready_to_go = self._is_ready()
        return self
    
    def read_parquet_from(self, path:str|Path) -> Self:
        self.df = pl.read_parquet(path)
        self.ready_to_go = self._is_ready()
        return self
    
    def from_pandas(self, df:pd.DataFrame) -> Self:
        print("!!! The input Pandas DataFrame will be emptied out after this call.")
        self.df = pl.from_pandas(df)
        df = df.iloc[0:0]
        self.ready_to_go = self._is_ready()
        return self
    # Add database support later

    def setup(self, data:Union[pl.DataFrame, pd.DataFrame, str, Path], target:str) -> Self:
        if isinstance(data, pl.DataFrame): 
            self.df = data
        elif isinstance(data, pd.DataFrame):
            print("!!! The input Pandas DataFrame will be emptied out after this call.")
            self.df = pl.from_pandas(data)
            data = data.iloc[0:0]
        else:
            try:
                if data.endswith(".csv"): # What about csv parameters? Use read_csv_from.
                    self.df = pl.read_csv(data)
                elif data.endswith(".parquet"):
                    self.df = pl.read_parquet(data)
                else:
                    raise NotImplementedError(f"The data/data path {data} is either not found or not supported at this time.")
            except Exception as e:
                print(e)
                self.ready_to_go = False

        self.target = target
        self.ready_to_go = self._is_ready()
        return self
    
    ### End of IO Section 
    ### Column removal section
    def set_null_removal(self, threshold:float) -> Self:
        if threshold > 1 or threshold <= 0:
            raise ValueError("Threshold for null removal must be between 0 and 1.")

        self.execution_plan.add_step(
            func = null_removal ,
            desc = BuiltinExecutions.NULL_REMOVAL.value.format(threshold*100),
            args = {"threshold":threshold}  
        )
        return self 
    
    def set_var_removal(self, threshold:float) -> Self:
        if threshold < 0:
            raise ValueError("Threshold for var removal must be positive.")
        
        self.execution_plan.add_step(
            func = var_removal,
            desc = BuiltinExecutions.VAR_REMOVAL.value.format(threshold),
            args = {"threshold":threshold},
        )
        return self
    
    def set_const_removal(self, include_null:bool=True) -> Self:
        
        self.execution_plan.add_step(
            func = constant_removal,
            desc = BuiltinExecutions.CONST_REMOVAL.value,
            args = {"threshold":include_null},
        )
        return self
    
    def set_regex_removal(self, pat:str, lowercase:bool=False) -> Self:

        self.execution_plan.add_step(
            func = regex_removal,
            desc = BuiltinExecutions.REGX_REMOVAL.value,
            args = {"pat":pat, "lowercase": lowercase},
        )
        return self
    
    def set_col_removal(self, cols:list[str]) -> Self:
        self.execution_plan.add_step(
            func = remove_if_exists,
            desc = BuiltinExecutions.COL_REMOVAL.value,
            args = {"to_drop":cols},
        )
        return self
        
    ### End of column removal section

    ### Scaling and Imputation
    def set_scaling(self, cols:list[str], strategy:ScalingStrategy.NORMALIZE, const:int=1) -> Self:
        # const only matters if startegy is constant
        self.execution_plan.add_step(
            func = scale,
            desc = BuiltinExecutions.SCALE.value.format(strategy),
            args = {"cols":cols, "strategy": strategy, "const":const},
            is_transformation = True
        )
        return self
    
    def set_impute(self, cols:list[str], strategy:ImputationStartegy.MEDIAN, const:int=1) -> Self:
        # const only matters if startegy is constant
        self.execution_plan.add_step(
            func = impute,
            desc = BuiltinExecutions.IMPUTE.value.format(strategy),
            args = {"cols":cols, "strategy": strategy, "const":const},
            is_transformation = True
        )
        return self
    
    ### End of Scaling and Imputation

    ### Encoding
    def set_ordinal_encoding(self, mapping:dict[str, dict[str,int]], default:int|None=None) -> Self:
        
        self.execution_plan.add_step(
            func = ordinal_encode,
            desc = BuiltinExecutions.ORDINAL_ENCODE.value,
            args = {"ordinal_mapping":mapping, "default": default},
            is_transformation = True
        )
        return self
    
    def set_ordinal_auto_encoding(self, cols:list[str], default:int|None=None) -> Self:
        
        self.execution_plan.add_step(
            func = ordinal_auto_encode,
            desc = BuiltinExecutions.ORDINAL_AUTO_ENCODE.value,
            args = {"ordinal_cols":cols, "default": default},
            is_transformation = True
        )
        return self
    
    def set_target_encoding(self, str_cols:list[str], min_samples_leaf:int=20, smoothing:int=10) -> Self:
        
        if self._is_ready():
            self.execution_plan.add_step(
                func = smooth_target_encode,
                desc = BuiltinExecutions.TARGET_ENCODE.value,
                args = {"target":self.target, "str_cols": str_cols, "min_samples_leaf":min_samples_leaf, "smoothing":smoothing},
                is_transformation = True
            )
            return self
        else:
            raise ValueError(f"The data frame and target must be set before setting target encoding.")
        
    def set_one_hot_encoding(self, one_hot_cols:list[str]) -> Self:
        pass
    
    def set_encode_by_record(self, rec:EncoderRecord) -> Self:
        
        self.execution_plan.add_step(
            func = encode_by,
            desc = BuiltinExecutions.ORDINAL_AUTO_ENCODE.value,
            args = {"rec":rec},
            is_transformation = True
        )
        return self
    
    def build(self) -> TransformedData: #(Use optional?)

        if self.ready_to_go:
            pass 
        else:
            print("Please make sure that:\n\t1. a dataframe input is correctly set.\n\t2. a target name is correct set.\n\t3. target exists as one of the columns of df.")
        



