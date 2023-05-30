from eda_utils import *
from dataclasses import dataclass
import polars as pl 
from typing import Self, Union, Optional, Any, Callable
from enum import Enum
from functools import partial
import os, time, json # Replace json by orjson?
from pathlib import Path

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

class _ExecChoice(Enum):
    NULL_REMOVAL = "Remove columns with more than {:.2f}% nulls."
    VAR_REMOVAL = "Remove columns with less than {} variance. (Not recommended.)"
    CONST_REMOVAL = "Remove columns that are constants."
    COL_REMOVAL = "Remove the columns {}."
    REGX_REMOVAL = "Remove all columns whose names satisfy the regex rule: {}."
    BINARY_ENCODE = "Encode {} into binary [0,1] values."
    ORDINAL_ENCODE = "Encode string values of {} into numbers with given mapping {}."
    ORDINAL_AUTO_ENCODE = "Encode string values of {} into numbers with inferred ordering."
    ONE_HOT_ENCODE = "Encode string values of {} by the one-hot-encoding technique. (No drop first option)."
    SCALE = "Scale {} using specified the {} scaling method."
    IMPUTE = "Impute {} using specified the {} imputation method."
    CHECKPOINT = "Unavailable for now."

@dataclass
class _ExecStep():
    name:_ExecChoice
    desc:str
    args:str # a json string representing the args. This will be useful when we "reconstruct" a pipeline from a blueprint.
    func:Callable

    def get_args(self) -> str:
        return self.args

    # def __init__(self, step:_ExecChoice, desc:str, func:Callable):
    #     self.name = step.name
    #     self.desc = desc
    #     self.func = func

@dataclass
class _ExecPlan():
    steps:list[_ExecStep] = []

    def add(self, step:_ExecStep) -> None:
        self.steps.append(step)

    def add_step(self, name:_ExecChoice, desc:str, func:Callable) -> None:
        self.steps.append(
            _ExecStep(name, desc, func)
        )

class TransformationBuilder:

    def __init__(self, project_name:str="my_proj") -> Self:
        self.df:pl.DataFrame = None
        self.target:str = ""
        self.project_name:str = project_name
        self.ready_to_go:bool = False
        self.execution_plan:_ExecPlan = _ExecPlan()
        return self
    
    def _is_ready(self) -> bool:
        return isinstance(self.df, pl.DataFrame) and not self.df.is_empty() and self.target in self.df.columns
    
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
    def read_csv_from(self, path:str|Path, csv_args:dict[str, Any]) -> Self:
        args = csv_args.copy()
        if "source" not in args:
            args["source"] = path

        self.df = pl.read_csv(**args)
        self.ready_to_go = self._is_ready()
        return self
    
    def read_parquet_from(self, path:str|Path) -> Self:
        self.df = pl.read_parquet(path)
        self.ready_to_go = self._is_ready()
        return self
    
    # Add database support later

    def setup(self, data:Union[pl.DataFrame, str, Path], target:str) -> Self:
        if isinstance(data, pl.DataFrame): 
            self.df = data 
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
            name = _ExecChoice.NULL_REMOVAL,
            desc = _ExecChoice.NULL_REMOVAL.value.format(threshold*100),
            args = json.dumps({"threshold":threshold}),
            func = partial(null_removal, threshold=threshold)        
        )
        return self 
    
    def set_var_removal(self, threshold:float) -> Self:
        if threshold < 0:
            raise ValueError("Threshold for var removal must be positive.")
        
        self.execution_plan.add_step(
            name = _ExecChoice.VAR_REMOVAL,
            desc = _ExecChoice.VAR_REMOVAL.value.format(threshold),
            args = json.dumps({"threshold":threshold}),
            func = partial(var_removal, threshold=threshold)            
        )
        return self
    
    def set_const_removal(self, include_null:bool=True) -> Self:
        
        self.execution_plan.add_step(
            name = _ExecChoice.CONST_REMOVAL,
            desc = _ExecChoice.CONST_REMOVAL.value,
            args = json.dumps({"threshold":include_null}),
            func = partial(constant_removal, include_null=include_null)            
        )
        return self
    
    def set_regx_removal(self, pat:str) -> Self:

        self.execution_plan.add_step(
            name = _ExecChoice.REGX_REMOVAL,
            desc = _ExecChoice.REGX_REMOVAL.value.format(pat),
            args = json.dumps({"pat":pat}),
            func = partial(regex_removal, pattern=pat)            
        )
        return self
        
    ### End of column removal section

    # def get_description(self) -> Self:
    #     self.execution_steps.append(Transformations.DESCRIBE)
    #     self.execution_funcs.append(describe)
    #     return self

    # def checkpoint(self, sample:float=1.0, sample_frac:float=-1.) -> Self:
    #     pass
    # Should have a better way of saving to local. For example, we hash the execution plan?
    #  
    
    def build(self) -> TransformedData: #(Use optional?)

        if self.ready_to_go:
            pass 
        else:
            print("Please make sure that:\n\t1. a dataframe input is correctly set.\n\t2. a target name is correct set.\n\t3. target exists as one of the columns of df.")
        



