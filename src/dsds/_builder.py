# from __future__ import annotations

# #######################################################################
# # ALL CODE HERE ARE FOR ARCHIVE PURPOSE 
# # WILL BE REMOVED ONCE THE NEW DEVELOPMENT BECOMES MORE STABLE
# #######################################################################

# from .type_alias import (
#     PolarsFrame
# )

# from .prescreen import (
#     remove_if_exists
#     , regex_removal
#     , var_removal
#     , null_removal
#     , unique_removal
#     , constant_removal
#     , date_removal
#     , non_numeric_removal
# )
# # from .eda_selection import *
# from .transform import (
#     FitRecord
#     , FitTransform
#     , ScalingStrategy
#     , ImputationStartegy
#     , scale
#     , impute
#     , binary_encode
#     , one_hot_encode
#     # , percentile_encode
#     , smooth_target_encode
#     , ordinal_encode
#     , ordinal_auto_encode
# )

# from dataclasses import dataclass
# import polars as pl
# from polars.type_aliases import FrameInitTypes
# import pandas as pd
# from typing import ParamSpec, Self, TypeVar, Optional, Any, Callable, Iterable, Concatenate
# from enum import Enum
# from pathlib import Path
# from time import perf_counter
# import orjson
# import inspect
# import re
# import logging
# import importlib
# import os

# T = TypeVar("T")
# P = ParamSpec("P")

# logger = logging.getLogger(__name__)

# ################################################################################################
# # WORK IN PROGRESS
# # 
# # For everyone who is reading this, my coding style is primarily influenced by the Primeagen,
# # some Rust conventions, and a little bit of functional style, although unfortunately we have to 
# # work with classes in Python.
# #
# # Terminologies: 
# # A ExecStep/transformation is canonical if the transformation on the dataset does not require
# # any information about the dataset to be performed or repeated.
# # e.g. raise the value of a column to the power of 2
# # e.g. remove columns with names a, b, c
# # They should be ExecSteps with is_fit = False and is_selector = False.
# # A canonical transformation should have the following signature:
# # Concatenate[pl.DataFrame, P] ---> pl.DataFrame, where
# # the output is just the transformed dataframe.
# # 
# # A non-canonical transformation is a transformation that requires information about the data to
# # be repeated.
# # e.g. impute (because we might need the mean/median of columns), scale, encoding... 
# # They should be ExecSteps with is_fit = True and is_selector = False.
# # A non-canonical transfromation should have the signature:
# # Concatenate[pl.DataFrame, P] ---> FitTransform, where
# # P = placeholder for other parameters
# # FitTransform is a data container that contains 
# # 1. transformed: the transformed dataframe
# # 2. mapping: a FitRecord that contains the necessary info about the data to repeat this fit
# #
# # A selector is a function that has the following signature:
# # Concatenate[pl.DataFrame, P] ---> list[str], where
# # P = placeholder for other parameters
# #
# ################################################################################################


# # This is just a builtin mapping for desc.
# class BuiltinExecs(Enum):
#     NULL_REMOVAL = "Remove columns with more than {:.2f}% nulls."
#     VAR_REMOVAL = "Remove columns with less than {} variance. (Not recommended.)"
#     CONST_REMOVAL = "Remove columns that are constants."
#     NON_NUMERIC_REMOVAL = "Remove all non-numeric columns."
#     UNIQUE_REMOVAL = "Remove columns that are like unique identifiers, e.g. with more than {:.2f}% unique values."
#     COL_REMOVAL = "Remove given if they exist in dataframe."
#     DATE_REMOVAL = "Remove columns that are inferred to be dates."
#     REGX_REMOVAL = "Remove all columns whose names satisfy the regex rule {}."
#     BINARY_ENCODE = "Encode given into binary [0,1] values."
#     ORDINAL_ENCODE = "Encode string values of given columns into numbers with given mapping."
#     ORDINAL_AUTO_ENCODE = "Encode string values of given columns into numbers with inferred ordering."
#     TARGET_ENCODE = "Encode string values using the target encoding algorithm."
#     ONE_HOT_ENCODE = "Encode string values of given columns by the one-hot-encoding technique. (No drop first option)."
#     PERCENTILE_ENCODE = "Encode a continuous column by percentiles."
#     SCALE = "Scale using specified the {} scaling method."
#     IMPUTE = "Impute using specified the {} imputation method."
#     CHECKPOINT = "Unavailable for now."
#     SELECT = "Select only the given columns."
#     LOWER = "Lower all column names in the incoming dataframe."

# @dataclass
# class ExecStep():
#     name:str
#     module:str|Path
#     desc:str = ""
#     args:Optional[dict[str, Any]] = None # None if it is record.
#     is_fit:bool = False
#     # Is this a function that produces a FitTransform output? 
#     # If so, the output is expected to be of type FitRecord (pl.DataFrame, FitRecord).
#     # If not, the output is expected to be of type pl.DataFrame. (Normal, natural output)

#     fit_name:Optional[str] = None # name of the FitRecord class (The class that inherits from FitRecord)
#     fit_module:Optional[str] = None # module where this FitRecord class belongs to
#     fit_record: Optional[FitRecord] = None # The actual content of the Record, will only be not none in blueprints.
#     is_selector:bool = False
#     is_custom:bool = False 

#     def get_args(self) -> str:
#         return self.args
    
#     def drop_args(self) -> Self:
#         self.args = None
    
#     def __str__(self) -> str:
#         text = f"Function: {self.name} | Module: {self.module} | Arguments:\n{self.args}\n"
#         text += f"Brief description: {self.desc}"
#         if self.is_fit:
#             text += "\nThis step is will fit and transform the data."
#         if self.is_custom:
#             text += "\nThis step is a user defined function."
#         return text

# @dataclass
# class ExecPlan():
#     steps:list[ExecStep]
#     target:str = ""

#     def __iter__(self) -> Iterable[ExecStep]:
#         return iter(self.steps)

#     def __str__(self) -> str:
#         if len(self.steps) > 0:
#             text = ""
#             for i, item in enumerate(self.steps):
#                 text += f"--- Step {i+1}: ---\n"
#                 text += str(item)
#                 text += "\n"
#             return text
#         else:
#             return "No step has been set."
        
#     def __len__(self) -> int:
#         return len(self.steps)
    
#     def clear(self) -> None:
#         self.steps.clear()

#     def is_empty(self) -> bool:
#         return len(self.steps) == 0

#     def add(self, step:ExecStep) -> None:
#         self.steps.append(step)

#     def popleft(self) -> Optional[ExecStep]:
#         return self.steps.pop(0)
    
#     def find(self, name:str) -> list[int]:
#         found:list[int] = []
#         regex = re.compile(name)
#         for i,step in enumerate(self.steps):
#             if regex.search(step.name) or regex.search(step.desc):
#                 found.append(i)

#         return found

#     def add_step(self
#         , func:Callable[Concatenate[pl.DataFrame, P], pl.DataFrame|FitTransform]
#         , desc:str
#         , args:dict[str, Any] # Technically is not Any, but Anything that can be serialized by orjson..
#         , is_fit:bool=False
#     ) -> None:

#         self.steps.append(
#             ExecStep(func.__name__, func.__module__, desc = desc, args = args
#                     , is_fit = is_fit, is_selector = False, is_custom = False)
#         )

#     def add_selector_step(self
#         , func:Callable[Concatenate[pl.DataFrame, P], list[str]]
#         , desc:str
#         , args:dict[str, Any]
#     ) -> None:
        
#         self.steps.append(
#             ExecStep(func.__name__, func.__module__, desc = desc, args = args, 
#                     is_fit = False, is_selector = True, is_custom = False)
#         )

#     def add_custom_step(self
#         , func:Callable[[pl.DataFrame, T], pl.DataFrame|FitTransform]
#         , desc:str
#         , args:dict[str, Any]
#         , is_fit:bool=False
#     ) -> None:
        
#         self.steps.append(
#             ExecStep(func.__name__, func.__module__, desc = desc, args = args, 
#                     is_fit = is_fit, is_selector = False, is_custom = True)
#         )

#     def add_custom_selector_step(self
#         , func:Callable[[pl.DataFrame, T], list[str]]
#         , desc:str
#         , args:dict[str, Any]
#     ) -> None:
        
#         self.steps.append(
#             ExecStep(func.__name__, func.__module__, desc = desc, args = args, 
#                     is_fit = False, is_selector = True, is_custom = True)
#         )

# def _select_cols(df:PolarsFrame, cols:list[str], target:str) -> PolarsFrame:

#     # Whether to select target or not depends on if df has a target column.
#     if target in df.columns and target not in cols:
#         cols.append(target)
#     elif target not in df.columns and target in cols:
#         cols.remove(target)

#     # don't check if c in cols belongs to df. Let it error.
#     return df.select(cols)

# def _lower_columns(df:PolarsFrame) -> PolarsFrame:
#     return df.rename({c: c.lower() for c in df.columns})

# def _rename(df:PolarsFrame, rename_dict:dict[str, str]) -> PolarsFrame:
#     return df.rename(rename_dict)

# # 1. The exact logic 
# # Pipebuilder -> ExecPlan (as execution_plan)
# #             -> ExecPlan (as _blueprint)
# # can be improved. Especially in execution_plan, we can totally just use the callable,
# # instead of importing it in the build process.
# # 
# # 2. Can we make output json smaller (slim down ExecStep)? Do we really need all the fields?
# # I want to avoid nesting. So it is flatter. Can we be more precise?
# #
# # 3. Better text representation
# #

# class PipeBuilder:

#     def __init__(self, target:str="", project_name:str="my_project"):

#         self.target:str = target
#         self.data:Optional[pl.LazyFrame] = None # This is always lazy
#         self.project_name:str = project_name
#         self._built:bool = False
#         self._execution_plan:ExecPlan = ExecPlan(steps=[])
#         self._blueprint:ExecPlan = ExecPlan(steps=[])
    
#     def __len__(self) -> int: # positive int
#         return len(self._execution_plan)
    
#     def __str__(self) -> str:
#         text = f"Project name: {self.project_name}\nTotal steps: {len(self)} |"
#         if self.target != "":
#             text += f" Target variable: {self.target}"
#         text += "\n"
#         if not self._execution_plan.is_empty() and not self._built:
#             text += str(self._execution_plan)
#             return text
#         elif self._built:
#             text += str(self._blueprint)
#             return text
#         return "Nothing to print."
    
#     ### I/O
#     def set_target(self, target:str) -> Self:
#         if target == "":
#             raise ValueError("Target cannot be empty string.")
        
#         if self.data is not None:
#             if target not in self.data.columns:
#                 raise ValueError("Target is not found in the dataframe.")
            
#         self.target = target
#         self._blueprint.target = target
#         return self
        
#     def set_data(self, df: FrameInitTypes|PolarsFrame) -> Self:
#         '''Set the data on which to "fit" the pipeline.'''
#         try:
#             if isinstance(df, pd.DataFrame):
#                 logger.warning("Found input to be a Pandas Dataframe. It will be converted to a Polars dataframe, "
#                             "and the original Pandas dataframe will be erased.")
                
#                 self.data = pl.from_pandas(df).lazy() # Keep it lazy for internal stuff.
#                 df = df.iloc[0:0]
#             elif isinstance(df, pl.DataFrame, pl.LazyFrame):
#                 self.data = df # Keep it lazy for internal stuff
#             else: # Try this..
#                 self.data = pl.DataFrame(df).lazy() # Keep it lazy for internal stuff
#         except Exception as e:
#             logger.error(e)

#         return Self
    
#     def set_data_and_target(self, df:Any, target:str) -> Self:
#         '''
#             df: Any data that is p
#         '''
#         _ = self.set_target(target)        
#         _ = self.set_data(df)
#         if target not in self.data.columns:
#             raise ValueError("Target is not found in the dataframe.")
        
#         self.target = target
#         return self 

#     # def from_csv(self, path:str|Path, **csv_args) -> Self:
#     #     '''
        
#     #     '''
#     #     try:
#     #         self.data = pl.read_csv(path, **csv_args)
#     #         return Self
#     #     except Exception as e:
#     #         logger.error(e)

#     # def from_parquet(self, path:str|Path, **parquet_args) -> Self:
#     #     '''
        
#     #     '''
#     #     try:
#     #         self.data = pl.read_parquet(path, **parquet_args)
#     #         return Self
#     #     except Exception as e:
#     #         logger.error(e)
    
#     # def from_json(self, path:str|Path, **json_args) -> Self:
#     #     '''
        
#     #     '''
#     #     try:
#     #         self.data = pl.read_json(path, **json_args)
#     #         return Self
#     #     except Exception as e:
#     #         logger.error(e)

#     # def from_excel(self, path:str|Path, **excel_args) -> Self:
#     #     '''
        
#     #     '''
#     #     try:
#     #         self.data = pl.read_json(path, **excel_args)
#     #         return Self
#     #     except Exception as e:
#     #         logger.error(e)
#     ### End of I/O
    
#     ### Miscellaneous
#     def show(self):
#         print(self)

#     def find(self, name:str) -> None:
#         found = self._execution_plan.find(name)
#         for i in found:
#             print(f"Step at index {i}:")
#             print(self._execution_plan.steps[i])

#     # Set, Remove, in a safe way.

#     def clear(self):
#         if self.data is not None:
#             self.data = self.data.clear()
#         self._execution_plan.clear()
#         self._blueprint.clear()
#         self._built = False

#     def select_cols(self, cols:list[str]) -> Self:
#         if self._is_ready():
#             self._execution_plan.add_step(
#                 func = _select_cols,
#                 desc = BuiltinExecs.SELECT.value,
#                 args = {"cols":cols, "target":self.target}
#             )
#             return self
#         else:
#             raise ValueError("Target must be set before setting column selections.")
    
#     def set_lower_cols(self) -> Self:
#         self._execution_plan.add_step(
#             func = _lower_columns,
#             desc = BuiltinExecs.LOWER.value,
#             args = {}
#         )
#         return self

#     ### End of Miscellaneous

#     ### Checks
#     def _is_ready(self) -> bool:
#         if self.data is None:
#             return False
        
#         return self.target != "" and self.target in self.data.columns
    
#     ### End of Checks.

#     ### Project meta data section
#     def set_project_name(self, project_name:str) -> Self:
#         if project_name == "":
#             raise ValueError("Project name cannot be empty.")
#         self.project_name = project_name
#         return self

#     ### End of project meta data section
    
#     ### Column removal section
#     def set_null_removal(self, threshold:float) -> Self:
#         if threshold > 1 or threshold <= 0:
#             raise ValueError("Threshold for null removal must be between 0 and 1.")

#         self._execution_plan.add_step(
#             func = null_removal ,
#             desc = BuiltinExecs.NULL_REMOVAL.value.format(threshold*100),
#             args = {"threshold":threshold}  
#         )
#         return self
    
#     def set_var_removal(self, threshold:float) -> Self:
#         if threshold <= 0:
#             raise ValueError("Threshold for var removal must be positive.")
        
#         if self._is_ready():
#             self._execution_plan.add_step(
#                 func = var_removal,
#                 desc = BuiltinExecs.VAR_REMOVAL.value.format(threshold),
#                 args = {"threshold":threshold, "target":self.target},
#             )
#             return self
#         else:
#             raise ValueError("Target must be set before setting var removal.")
    
#     def set_const_removal(self, include_null:bool=True) -> Self:
        
#         self._execution_plan.add_step(
#             func = constant_removal,
#             desc = BuiltinExecs.CONST_REMOVAL.value,
#             args = {"include_null":include_null},
#         )
#         return self
    
#     def set_unique_removal(self, threshold:float=0.9) -> Self:
#         if threshold > 1 or threshold <= 0:
#             raise ValueError("Threshold for unique removal must be between 0 and 1.")

#         self._execution_plan.add_step(
#             func = unique_removal ,
#             desc = BuiltinExecs.UNIQUE_REMOVAL.value.format(threshold*100),
#             args = {"threshold":threshold}  
#         )
#         return self 
    
#     def set_regex_removal(self, pat:str, lowercase:bool=False) -> Self:
#         '''Removes columns that satisfies the regex rule. May optionally only check on lowercased column names.'''
#         description = BuiltinExecs.REGX_REMOVAL.value.format(pat)
#         if lowercase:
#             description += ". Everything will be lowercased."

#         self._execution_plan.add_step(
#             func = regex_removal,
#             desc = description,
#             args = {"pat":pat, "lowercase": lowercase},
#         )
#         return self
    
#     def set_col_removal(self, cols:list[str]) -> Self:
#         '''Removes the given columns.'''
#         self._execution_plan.add_step(
#             func = remove_if_exists,
#             desc = BuiltinExecs.COL_REMOVAL.value,
#             args = {"to_drop":cols},
#         )
#         return self
    
#     def set_date_removal(self) -> Self:
#         '''Removes columns that are inferred to be dates.'''
#         self._execution_plan.add_step(
#             func = date_removal,
#             desc = BuiltinExecs.DATE_REMOVAL.value,
#             args = {}
#         )
#         return self
    
#     def set_non_numeric_removal(self) -> Self:
#         self._execution_plan.add_step(
#             func = non_numeric_removal ,
#             desc = BuiltinExecs.NON_NUMERIC_REMOVAL.value,
#             args = {}
#         )
#         return self 
        
#     ### End of column removal section

#     ### Scaling and Imputation
#     def set_scaling(self, cols:list[str]
#         , strategy:ScalingStrategy=ScalingStrategy.NORMALIZE
#         , const:int=1) -> Self:
#         # const only matters if startegy is constant
#         self._execution_plan.add_step(
#             func = scale,
#             desc = BuiltinExecs.SCALE.value.format(strategy),
#             args = {"cols":cols, "strategy": strategy, "const":const},
#             is_fit = True
#         )
#         return self
    
#     def set_impute(self, cols:list[str]
#         , strategy:ImputationStartegy=ImputationStartegy.MEDIAN
#         , const:int=1) -> Self:
        
#         # const only matters if startegy is constant
#         self._execution_plan.add_step(
#             func = impute,
#             desc = BuiltinExecs.IMPUTE.value.format(strategy),
#             args = {"cols":cols, "strategy": strategy, "const":const},
#             is_fit = True
#         )
#         return self
    
#     ### End of Scaling and Imputation

#     ### Encoding
#     def set_binary_encoding(self, cols:Optional[list[str]]=None) -> Self:
        
#         if cols:
#             description = BuiltinExecs.BINARY_ENCODE.value
#         else:
#             description = "Automatically detect binary columns and turn them into [0,1] values by their order."

#         self._execution_plan.add_step(
#             func = binary_encode,
#             desc = description,
#             args = {"cols":cols},
#             is_fit = True
#         )
#         return self

#     def set_ordinal_encoding(self, mapping:dict[str, dict[str,int]], default:Optional[int]=None) -> Self:
        
#         self._execution_plan.add_step(
#             func = ordinal_encode,
#             desc = BuiltinExecs.ORDINAL_ENCODE.value,
#             args = {"ordinal_mapping":mapping, "default": default},
#             is_fit = True
#         )
#         return self
    
#     def set_ordinal_auto_encoding(self, cols:list[str], default:Optional[int]=None) -> Self:
        
#         self._execution_plan.add_step(
#             func = ordinal_auto_encode,
#             desc = BuiltinExecs.ORDINAL_AUTO_ENCODE.value,
#             args = {"cols":cols, "default": default},
#             is_fit = True
#         )
#         return self
    
#     def set_target_encoding(self, cols:list[str], min_samples_leaf:int=20, smoothing:int=10) -> Self:
        
#         if self._is_ready():
#             self._execution_plan.add_step(
#                 func = smooth_target_encode,
#                 desc = BuiltinExecs.TARGET_ENCODE.value,
#                 args = {"target":self.target, "cols": cols, "min_samples_leaf":min_samples_leaf
#                         , "smoothing":smoothing},
#                 is_fit = True
#             )
#             return self
#         else:
#             raise ValueError("The target must be set before target encoding.")
        
#     def set_one_hot_encoding(self, cols:Optional[list[str]]=None, separator:str="_") -> Self:
        
#         if cols:
#             description = BuiltinExecs.ORDINAL_AUTO_ENCODE.value
#         else:
#             description = "Automatically detect string columns and one-hot encode them."

#         self._execution_plan.add_step(
#             func = one_hot_encode,
#             desc = description,
#             args = {"cols":cols, "separator": separator},
#             is_fit = True
#         )
#         return self
    
#     # def set_percentile_encoding(self, cols:list[str]) -> Self:

#     #     self._execution_plan.add_step(
#     #         func = percentile_encode,
#     #         desc = BuiltinExecs.PERCENTILE_ENCODE.value,
#     #         args = {"cols":cols},
#     #         is_fit = True
#     #     )
#     #     return self
    
#     ### End of Encoding Section

#     ### Custom Actions

#     # Step and FitTransforms are differentiated in order to prevent error.
#     # A normal step is a function that takes in a dataframe, some other args, and 
#     # outputs a dataframe. Use add_custom_step for this case.
#     #
#     # A FitTransform is essentially a function that outputs the type FitTransform.
#     # This means that the transformation is dependent on information about the dataset (self.data),
#     # If this is the case, use add_custom_fit_transform

#     def add_custom_step(self
#         , func:Callable[Concatenate[PolarsFrame, P], PolarsFrame]
#         , desc:str
#         , args:dict[str, Any]
#         ) -> Self:
#         '''A normal step is a function that takes in a dataframe, some other args, and outputs a dataframe.'''
#         if "return" not in func.__annotations__:
#             logger.info("It is highly recommended that the custom step returns " 
#                         "a Polars dataframe. If this returns a TransformationResult, use add_custom_transform instead.")

#         self._execution_plan.add_custom_step(
#             func = func,
#             desc = desc,
#             args = args,
#             is_fit = False
#         )

#         return self
    
#     def add_selector(self
#         , func:Callable[Concatenate[PolarsFrame, P], list[str]]
#         , desc:str
#         , args:dict[str, Any]
#         ) -> Self:

#         if str(inspect.signature(func).return_annotation) != "list[str]":
#             raise TypeError("A selector function must explicitly provide a list[str] return type.")

#         if self._is_ready():
#             if "target" not in args:
#                 args["target"] = self.target
#             self._execution_plan.add_selector_step(
#                 func = func,
#                 desc = desc,
#                 args = args
#             )
#             return self
#         else:
#             raise ValueError("Selectors can only be queued after df and target are set.")
    
#     def add_custom_selector(self
#         , func:Callable[Concatenate[PolarsFrame, P], list[str]]
#         , desc:str
#         , args:dict[str, Any]
#         ) -> Self:
#         '''A normal step is a function that takes in a dataframe, some other args, and outputs a dataframe.'''
#         if str(inspect.signature(func).return_annotation) != "list[str]":
#             raise TypeError("A selector function must explicitly provide a list[str] return type.")

#         if self._is_ready():
#             if "target" not in args:
#                 args["target"] = self.target
#             self._execution_plan.add_custom_selector_step(
#                 func = func,
#                 desc = desc,
#                 args = args
#             )
#             return self
#         else:
#             raise ValueError("Selectors can only be queued after df and target are set.")

#     def add_custom_fit_transform(self
#         , func:Callable[Concatenate[PolarsFrame, P], FitTransform]
#         , desc:str
#         , args:dict[str, Any]
#         ) -> Self:
#         '''A FitTransform is essentially a function that outputs the type FitTransform.
#         This means that the transformation is dependent on information about the dataset (self.data),
#         '''

#         if "return" not in func.__annotations__:
#             logger.info("It is highly recommended that the custom transform returns "
#                         "a type that inherits from FitTransform in the transform.py module."
#                         "If this returns a pl.DataFrame, use add_custom_step instead.")

#         self._execution_plan.add_custom_step(
#             func = func,
#             desc = desc,
#             args = args,
#             is_fit = True
#         )

#         return self

#     ### End of Custom Actions


#     def _process_fit_in_build(self, step:ExecStep) -> None:

#         apply_transf:Callable[Concatenate[PolarsFrame, P], FitTransform] 
#         apply_transf = getattr(importlib.import_module(step.module), step.name)
#         rec: FitRecord
#         self.data, rec = self.data.pipe(apply_transf, **step.args)
#         new_step = ExecStep(
#             name = step.name,
#             module = "N/A",
#             desc = step.desc, # 
#             # args = step.args, # don't need args when we apply (transform)
#             is_fit = True,
#             fit_name = type(rec).__name__,
#             fit_module = rec.__module__,
#             fit_record = rec,
#             is_custom = step.is_custom
#         )
#         self._blueprint.add(new_step)

#     def _process_selector_in_build(self, step:ExecStep) -> None:

#         selector:Callable[Concatenate[pl.DataFrame, P], list[str]]
#         selector = getattr(importlib.import_module(step.module), step.name)
#         selected_cols:list[str] = selector(self.data, **step.args)
#         to_select = selected_cols.copy()
#         if self.target not in selected_cols:
#             to_select.append(self.target)
        
#         # In this stage, target should be in to_select and in df.columns.
#         self.data = self.data.pipe(_select_cols, to_select, self.target)

#         logger.info(f"The following features are kept: {selected_cols[:10]} + ... Only showing top 10.")
#         self._blueprint.add_step(
#             func = _select_cols,
#             desc = step.desc,
#             args = {"cols": selected_cols, "target": self.target}
#         )


#     # RETHINK how to build.
#     # Maybe we should cast df to lazy, then use Polars's Pipe. But how to save Polars's Pipe?
#     def build_2(self) -> pl.DataFrame:
#         pass

#     def build(self) -> pl.DataFrame:
#         '''Build according to the steps.
#             Arguments:
#                 df: Another chance to set input df if it has not been set.

#             Returns:
#                 A dataframe.
        
#         '''
#         n = len(self._execution_plan)
#         logger.info(f"Starting to build. Total steps: {n}.")
#         if not self._is_ready():
#             raise ValueError(f"Dataframe is not set properly, or the target {self.target} is not found "
#                              "in the dataframe. Cannot build without target.")
        
#         if self._built:
#             logger.warning("The PipeBuilder is built once already. It is not intended to be built again. "
#                            "To avoid unexpected behavior, construct a new pipe only after calling .clear().")
        
#         i = 0
#         # Todo! If something failed, save a backup dataframe to a temp folder.
#         while not self._execution_plan.is_empty():
#             i += 1
#             step = self._execution_plan.popleft()
#             logger.info(f"|{i}/{n}|: Step: {step.name} | is_fit: {step.is_fit} | is_selector: {step.is_selector}")
#             start = perf_counter()
#             success = True

#             if step.is_selector:
#                 self._process_selector_in_build(step)
#             elif step.is_fit:
#                 self._process_fit_in_build(step)
#             else: # Regular, canonical steps.
#                 apply_func:Callable[Concatenate[PolarsFrame, P], PolarsFrame]  
#                 apply_func = getattr(importlib.import_module(step.module), step.name)
#                 self.data = self.data.pipe(apply_func, **step.args)
#                 self._blueprint.add(step)

#             end = perf_counter()
#             logger.info(f"|{i}/{n}|: Finished in {end-start:.2f}s | Success: {success}")

#         logger.info("Build success. A blueprint has been built and can be viewed by calling .blueprint(), "
#                     "and can be saved as a json by calling .write()")

#         self._built = True
#         return self.data
    
#     # Rename this in the future?
#     def apply(self, df:pl.DataFrame|pd.DataFrame) -> pl.DataFrame:
#         if not self._built:
#             raise ValueError("The builder must be built before applying it to new datasets.")
#         try:
#             if isinstance(df, pd.DataFrame):
#                 logger.warning("Found input to be a Pandas dataframe. Turning it into a Polars dataframe.")
#                 try:
#                     input_df:pl.DataFrame = pl.from_pandas(df)
#                 except Exception as e:
#                     logger.error(e)
#             elif isinstance(df, pl.DataFrame):
#                 input_df:pl.DataFrame = df
#             else:
#                 input_df:pl.DataFrame = pl.DataFrame(df)
#         except Exception as e:
#             logger.error(e)

#         n = len(self._blueprint)
#         step:ExecStep
#         for i, step in enumerate(self._blueprint):
#             logger.info(f"|{i+1}/{n}|: Performing Step: {step.name} | is_fit: {step.is_fit}")
#             start = perf_counter()
#             success = True
#             if step.is_fit:
#                 try:
#                     rec:FitRecord = step.fit_record
#                     input_df = input_df.pipe(rec.transform)
#                 except Exception as e:
#                     success = False
#                     logger.error(e)
#             else:
#                 apply_func:Callable[Concatenate[pl.DataFrame, P], pl.DataFrame] 
#                 apply_func = getattr(importlib.import_module(step.module), step.name)
#                 input_df = input_df.pipe(apply_func, **step.args)

#             end = perf_counter()
#             logger.info(f"|{i+1}/{n}|: Finished in {end-start:.2f}s | Success: {success}")
        
#         return input_df

#     def blueprint(self):
#         return print(self._blueprint)
        
#     def write(self, name:str="") -> None:
#         if self._blueprint.is_empty():
#             logger.warning("Blueprint is empty. Nothing is done.")
#             return

#         directory = "./blueprints/"
#         if name == "":
#             name += self.project_name + ".json"
#             logger.info(f"No name is specified, using project name ({name}) as default.")
#         else:
#             if not name.endswith(".json"):
#                 name += ".json"

#         if not os.path.isdir(directory):
#             logger.info("Local ./blueprints/ directory is not found. It will be created.")
#             os.mkdir(directory)
#         try:
#             destination = directory+name
#             with open(destination, "wb") as f:
#                 data = orjson.dumps(self._blueprint, option=orjson.OPT_NON_STR_KEYS|orjson.OPT_SERIALIZE_NUMPY)
#                 f.write(data)

#             logger.info(f"Successfully saved to {destination}.")
#         except Exception as e:
#             logger.error(e)

#     def from_blueprint(self, input_file:str|Path|bytes) -> Self:
#         logger.info("Reading from a blueprint. The builder will reset itself.")
#         self.clear()
#         try:
#             if isinstance(input_file, bytes):
#                 data = orjson.loads(input_file)
#             else:
#                 f = open(input_file, "rb")
#                 data = orjson.loads(f.read())
#                 f.close()
            
#             steps:list[dict[str, Any]] = data["steps"]
#             self.target = data["target"]
#             for s in steps:
#                 if s["is_fit"]: # Need to recreate TransformRecord objects from dict
#                     name = s.get("fit_name", None)
#                     module = s.get("fit_module", None)
#                     record = s.get("fit_record", None)
#                     if (name is not None) or (module is not None) or (record is not None):
#                         if (name is None) or (module is None) or (record is None):
#                             raise ValueError(f"Something went wrong with the FitRecord: {s['name']}. "
#                                              "All of fit_name, fit_module and fit_record fields must not be null.")
#                     # Get the class the FitRecord belongs to.
#                     c = getattr(importlib.import_module(module), name)
#                     # Create an instance of c
#                     rec = c(**record) # Turn this json into a real Python object.
#                     s["fit_record"] = rec # Set "fit_record" field to be the object, not the dict.

#                 self._blueprint.add(ExecStep(**s))

#             self._built = True
#             logger.info("Successfully read from a blueprint.")
#         except Exception as e:
#             logger.error(e)

#         return self
#     ### End of Building section