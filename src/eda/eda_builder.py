from __future__ import annotations

from .eda_prescreen import *
from .eda_selection import *
from .eda_transformations import *
from dataclasses import dataclass, field
import polars as pl
import pandas as pd
from typing import ParamSpec, Self, TypeVar, Union, Optional, Any, Callable, Iterable
from enum import Enum
# from functools import partial
# import os, time, json 
from pathlib import Path
import logging

T = TypeVar("T")
P = ParamSpec("P")
logger = logging.getLogger(__name__)
################################################################################################
# WORK IN PROGRESS
################################################################################################

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
    UNIQUE_REMOVAL = "Remove columns that are like unique identifiers, e.g. with more than {:.2f}% unique values."
    COL_REMOVAL = "Remove given if they exist in dataframe."
    REGX_REMOVAL = "Remove all columns whose names satisfy the regex rule {}."
    BINARY_ENCODE = "Encode given into binary [0,1] values."
    ORDINAL_ENCODE = "Encode string values of given columns into numbers with given mapping."
    ORDINAL_AUTO_ENCODE = "Encode string values of given columns into numbers with inferred ordering."
    TARGET_ENCODE = "Encode string values using the target encoding algorithm."
    ONE_HOT_ENCODE = "Encode string values of given columns by the one-hot-encoding technique. (No drop first option)."
    PERCENTILE_ENCODE = "Encode a continuous column by percentiles."
    SCALE = "Scale using specified the {} scaling method."
    IMPUTE = "Impute using specified the {} imputation method."
    CHECKPOINT = "Unavailable for now."
    SELECT = "Select only the given columns."

@dataclass
class ExecStep():
    name:str
    module:str|Path|None
    desc:str
    args:dict[str, Any]
    is_transformation:bool = False
    # Is this a transfromation call? 
    # If so, the output is expected to be of type TransformationResult.
    # If not, the output is expected to be of type pl.DataFrame.
    is_custom:bool = False 

    def get_args(self) -> str:
        return self.args
    
    def __str__(self) -> str:
        text = f"Function: {self.name} | Module: {self.module} | Arguments:\n{self.args}\n"
        text += f"Brief description: {self.desc}"
        if self.is_transformation:
            text += "\nThis step is a transformation step."
        if self.is_custom:
            text += "\nThis step is a user defined function."
        return text

@dataclass
class ExecPlan():
    _steps:list[ExecStep] = field(default_factory=list)

    def __iter__(self) -> Iterable:
        return iter(self._steps)

    def __str__(self) -> str:
        if len(self._steps) > 0:
            text = ""
            for i, item in enumerate(self._steps):
                text += f"--- Step {i+1}: ---\n"
                text += str(item)
                text += "\n"
            return text
        else:
            return "No step has been set."
        
    def __len__(self) -> int:
        return len(self._steps)
    
    def clear(self) -> None:
        self._steps.clear()

    def is_empty(self) -> bool:
        return len(self._steps) == 0

    def add(self, step:ExecStep) -> None:
        self._steps.append(step)

    def add_step(self
        , func:Callable[[pl.DataFrame, T], pl.DataFrame|TransformationResult]
        , desc:str
        , args:dict[str, Any] # Technically is not Any, but Anything that can be serialized by orjson..
        , is_transf:bool=False) -> None:
        
        self._steps.append(
            ExecStep(func.__name__, func.__module__, desc, args, is_transf, False)
        )

    def add_custom_step(self
        , func:Callable[[pl.DataFrame, T], pl.DataFrame|TransformationResult]
        , desc:str
        , args:dict[str, Any]
        , is_transf:bool=False) -> None:
        
        self._steps.append(
            ExecStep(func.__name__, func.__module__, desc, args, is_transf, True)
        )

def _select_cols(df:pl.DataFrame, cols:list[str]) -> pl.DataFrame:
    return df.select(cols)

class TransformationBuilder:

    def __init__(self, target:str, project_name:str="my_project") -> Self:
        if target == "":
            raise ValueError("Target cannot be empty string. Please rename it if it is the case.")

        self.target = target
        self.project_name:str = project_name
        self._built:bool = False
        self._execution_plan:ExecPlan = ExecPlan()
        self._blueprint:ExecPlan = ExecPlan()
        return self
    
    def __len__(self) -> int: # positive int
        return len(self._execution_plan)
    
    def __str__(self) -> str:
        if not self._execution_plan.is_empty():
            text = f"Project name: {self.project_name}"
            f"\nTotal steps: {len(self)} | Ready to build: {self._is_ready()} | Is built: {self._built}\n"
            if self.target != "":
                text += f"Target variable: {self.target}"
            text += "\n"
            text += str(self._execution_plan)
            return text
        return "The current builder has no execution plan."
    
    ### Miscellaneous
    def show(self):
        print(self)

    def clear(self):
        self.target = ""
        self._execution_plan.clear()
        self._blueprint.clear()
        self._built = False

    def select_cols(self, cols:list[str]) -> Self:
        self._execution_plan.add_step(
            func = _select_cols,
            desc = BuiltinExecutions.SELECT.value,
            args = {"cols":cols}
        )
        return self

    ### End of Miscellaneous

    ### Checks
    def _is_ready(self) -> bool:
        return self.target != ""
    
    ### End of Checks.

    ### Project meta data section
    def set_project_name(self, project_name:str) -> Self:
        if project_name == "":
            raise ValueError("Project name cannot be empty.")
        self.project_name = project_name
        return self

    ### End of project meta data section
    
    ### IO Section 
    def from_blueprint(self) -> Self:
        pass

    ### End of IO Section 
    ### Column removal section
    def set_null_removal(self, threshold:float) -> Self:
        if threshold > 1 or threshold <= 0:
            raise ValueError("Threshold for null removal must be between 0 and 1.")

        self._execution_plan.add_step(
            func = null_removal ,
            desc = BuiltinExecutions.NULL_REMOVAL.value.format(threshold*100),
            args = {"threshold":threshold}  
        )
        return self 
    
    def set_var_removal(self, threshold:float) -> Self:
        if threshold <= 0:
            raise ValueError("Threshold for var removal must be positive.")
        
        self._execution_plan.add_step(
            func = var_removal,
            desc = BuiltinExecutions.VAR_REMOVAL.value.format(threshold),
            args = {"threshold":threshold},
        )
        return self
    
    def set_const_removal(self, include_null:bool=True) -> Self:
        
        self._execution_plan.add_step(
            func = constant_removal,
            desc = BuiltinExecutions.CONST_REMOVAL.value,
            args = {"threshold":include_null},
        )
        return self
    
    def set_unique_removal(self, threshold:float=0.9) -> Self:
        if threshold > 1 or threshold <= 0:
            raise ValueError("Threshold for unique removal must be between 0 and 1.")

        self._execution_plan.add_step(
            func = unique_removal ,
            desc = BuiltinExecutions.UNIQUE_REMOVAL.value.format(threshold*100),
            args = {"threshold":threshold}  
        )
        return self 
    
    def set_regex_removal(self, pat:str, lowercase:bool=False) -> Self:

        description = BuiltinExecutions.REGX_REMOVAL.value.format(pat)
        if lowercase:
            description += ". Everything will be lowercased."

        self._execution_plan.add_step(
            func = regex_removal,
            desc = description,
            args = {"pat":pat, "lowercase": lowercase},
        )
        return self
    
    def set_col_removal(self, cols:list[str]) -> Self:
        self._execution_plan.add_step(
            func = remove_if_exists,
            desc = BuiltinExecutions.COL_REMOVAL.value,
            args = {"to_drop":cols},
        )
        return self
        
    ### End of column removal section

    ### Scaling and Imputation
    def set_scaling(self, cols:list[str], strategy:ScalingStrategy.NORMALIZE, const:int=1) -> Self:
        # const only matters if startegy is constant
        self._execution_plan.add_step(
            func = scale,
            desc = BuiltinExecutions.SCALE.value.format(strategy),
            args = {"cols":cols, "strategy": strategy, "const":const},
            is_transformation = True
        )
        return self
    
    def set_impute(self, cols:list[str], strategy:ImputationStartegy.MEDIAN, const:int=1) -> Self:
        # const only matters if startegy is constant
        self._execution_plan.add_step(
            func = impute,
            desc = BuiltinExecutions.IMPUTE.value.format(strategy),
            args = {"cols":cols, "strategy": strategy, "const":const},
            is_transformation = True
        )
        return self
    
    ### End of Scaling and Imputation

    ### Encoding
    def set_binary_encoding(self, bin_cols:Optional[list[str]]=None) -> Self:
        
        if bin_cols:
            description = BuiltinExecutions.BINARY_ENCODE.value
        else:
            description = "Automatically detect binary columns and turn them into [0,1] values according to their order."

        self._execution_plan.add_step(
            func = binary_encode,
            desc = description,
            args = {"binary_cols":bin_cols},
            is_transformation = True
        )
        return self

    def set_ordinal_encoding(self, mapping:dict[str, dict[str,int]], default:Optional[int]=None) -> Self:
        
        self._execution_plan.add_step(
            func = ordinal_encode,
            desc = BuiltinExecutions.ORDINAL_ENCODE.value,
            args = {"ordinal_mapping":mapping, "default": default},
            is_transformation = True
        )
        return self
    
    def set_ordinal_auto_encoding(self, cols:list[str], default:Optional[int]=None) -> Self:
        
        self._execution_plan.add_step(
            func = ordinal_auto_encode,
            desc = BuiltinExecutions.ORDINAL_AUTO_ENCODE.value,
            args = {"ordinal_cols":cols, "default": default},
            is_transformation = True
        )
        return self
    
    def set_target_encoding(self, str_cols:list[str], min_samples_leaf:int=20, smoothing:int=10) -> Self:
        
        if self._is_ready():
            self._execution_plan.add_step(
                func = smooth_target_encode,
                desc = BuiltinExecutions.TARGET_ENCODE.value,
                args = {"target":self.target, "str_cols": str_cols, "min_samples_leaf":min_samples_leaf, "smoothing":smoothing},
                is_transformation = True
            )
            return self
        else:
            raise ValueError(f"The data frame and target must be set before setting target encoding.")
        
    def set_one_hot_encoding(self, one_hot_cols:Optional[list[str]]=None, separator:str="_") -> Self:
        
        if one_hot_cols:
            description = BuiltinExecutions.ORDINAL_AUTO_ENCODE.value
        else:
            description = "Automatically detect string columns and one-hot encode them."

        self._execution_plan.add_step(
            func = one_hot_encode,
            desc = description,
            args = {"one_hot_cols":one_hot_cols, "separator": separator},
            is_transformation = True
        )
        return self
    
    def set_percentile_encoding(self, num_cols:list[str]) -> Self:
        
        self._execution_plan.add_step(
            func = percentile_encode,
            desc = BuiltinExecutions.PERCENTILE_ENCODE.value,
            args = {"num_cols":num_cols},
            is_transformation = True
        )
        return self
    
    ### End of Encoding Section

    ### Custom Actions
    def add_custom_action(self
        , func:Callable[[pl.DataFrame, T], pl.DataFrame|TransformationResult]
        , desc:str
        , args:dict[str, Any]
        , is_transformation:bool) -> Self:

        if is_transformation:
            logging.info("It is highly recommended that the custom function returns a type that inherits from the TransformationRecord class.")
        self._execution_plan.add_custom_step(
            func = func,
            desc = desc,
            args = args,
            is_transformation = is_transformation
        )

        return self

    ### End of Custom Actions

    def fit(self, X, y) -> Self:
        pass

    def build(self, df:pl.DataFrame|pd.DataFrame) -> Self:
        if not self._is_ready():
            logger.warning("Not ready to build. Please make sure that both the dataframe and the target are set up properly.")
            return None
        if self._built:
            logger.warning("The builder has already been built once. Please run")


        try:
            pass

        except Exception as e:
            logger.error(e)
            return None
        
        
    def write_blueprint(self, path:str|Path) -> None:
        if self._blueprint.is_empty():
            pass

        



