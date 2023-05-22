from eda_utils import *
from dataclasses import dataclass
import polars as pl 
from typing import Self
from enum import Enum
from functools import partial
import os, time

class TransformedData:
    
    def __init__(self):
        self.df:pl.DataFrame = None
        self.target:str = ""
        self.log:list[str] = []
        self.mappings:list[pl.DataFrame] = []

class Transformations(Enum):
    DESCRIBE = 0
    NULL_REMOVAL = 1
    VAR_REMOVAL = 2

class TransformationBuilder:

    def __init__(self, df:pl.DataFrame = None, target:str = "", project_name:str="my_proj") -> Self:
        self.df:pl.DataFrame = df 
        self.target:str = target
        self.project_name:str = project_name
        self.execution_steps:list[Transformations] = []
        self.execution_funcs: list = []
        return self 

    def set_data_and_target(self, df:pl.DataFrame, target:str) -> Self:
        self.df = df 
        self.target = target
        return self
    
    def set_target(self, target:str) -> Self: 
        self.target = target 
        return self 
    
    def set_data(self, data:pl.DataFrame) -> Self:
        self.df = data 
        return self 

    def set_null_removal(self, threshold:float) -> Self:
        self.execution_steps.append(Transformations.NULL_REMOVAL)
        func = partial(null_removal, threshold=threshold)
        self.execution_funcs.append(func)
        return self 
    
    def set_var_removal(self, threshold:float) -> Self:
        self.execution_steps.append(Transformations.VAR_REMOVAL)
        func = partial(var_removal, threshold=threshold)
        self.execution_funcs.append(func)
        return self
    
    def get_description(self) -> Self:
        self.execution_steps.append(Transformations.DESCRIBE)
        self.execution_funcs.append(describe)
        return self
    
    def _checkpoint(self, sample_amt:int=-1, sample_frac:float=-1.):
        
        output_path = "./.checkpoint/"
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        final_path = output_path + self.project_name + "_checkpoint_"
        all_files = list(filter(lambda f: f.startswith(final_path), os.listdir(output_path)))
        if len(all_files) == 0:
            final_path += "0.csv"
        else:
            all_files.sort()
            last_checkpoint:str = all_files[-1].replace(".csv", "")
            last_checkpoint_num:int = int(last_checkpoint.split("_")[-1]) + 1
            final_path += (str(last_checkpoint_num) + ".csv")

        if sample_amt <= 0:
            self.df.write_csv(final_path)
        else:
            # If both amount and frac are set, prioritize frac
            if sample_frac > 0.0 and sample_frac < 1.0: # frac is set
                self.df.sample(fraction=sample_frac).write_csv(final_path)
            else: # frac is not set, or is just a non-sensical value
                self.df.sample(n=sample_amt).write_csv(final_path)

        return self 

    def checkpoint(self, sample:float=1.0) -> Self:
        pass
    
    def build(self) -> TransformedData: #(Use optional?)

        if isinstance(self.df, None) or self.target == "":
            raise ValueError("self.df cannot be None when building and self.target must be set.")
        



