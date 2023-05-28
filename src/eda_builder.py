from eda_utils import *
from dataclasses import dataclass
import polars as pl 
from typing import Self, Union, Optional, Any
from enum import Enum
from functools import partial
import os, time
from pathlib import Path

class TransformedData:
    
    def __init__(self):
        self.df:pl.DataFrame = None
        self.target:str = ""
        self.log:list[str] = []
        self.mappings:list[pl.DataFrame] = []

# We need this abstraction for the following reasons
# 1. In the future, I want to add "checkpoint" (save to local)
# But if something is saved to local before, I want the plan to 
# notice it and read directly to local and skip the "checkpointed" steps.
# 2. Better organization the execution.
#

class _ExecutionPlan():
    
    def __init__(self):
        pass

class TransformationBuilder:

    def __init__(self, project_name:str="my_proj") -> Self:
        self.df:pl.DataFrame = None
        self.target:str = ""
        self.project_name:str = project_name
        self.ready_to_go:bool = False
        # These will be changed soon once I figured out the data structure for ExecutionPlan
        self.execution_steps:list = []
        self.execution_funcs: list = []
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

    def set_null_removal(self, threshold:float) -> Self:
        
        func = partial(null_removal, threshold=threshold)
        self.execution_funcs.append(func)
        return self 
    
    def set_var_removal(self, threshold:float) -> Self:
        
        func = partial(var_removal, threshold=threshold)
        self.execution_funcs.append(func)
        return self
    
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
        



