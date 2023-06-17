from __future__ import annotations

from .prescreen import (
    remove_if_exists
    , regex_removal
    , var_removal
    , null_removal
    , unique_removal
    , constant_removal
    , date_removal
)
# from .eda_selection import *
from .transform import (
    TransformationResult
    , TransformationRecord
    , ScalingStrategy
    , ImputationStartegy
    , scale
    , impute
    , binary_encode
    , one_hot_encode
    , percentile_encode
    , smooth_target_encode
    , ordinal_encode
    , ordinal_auto_encode
)

from dataclasses import dataclass
import polars as pl
import pandas as pd
from typing import ParamSpec, Self, TypeVar, Optional, Any, Callable, Iterable, Concatenate
from enum import Enum
from pathlib import Path
from time import perf_counter
import orjson
import logging
import importlib
import os

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
    DATE_REMOVAL = "Remove columns that are inferred to be dates."
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
    LOWER = "Lower all column names in the incoming dataframe."

@dataclass
class ExecStep():
    name:str
    module:str|Path
    desc:str = ""
    args:Optional[dict[str, Any]] = None # None if it is record.
    is_transform:bool = False
    # Is this a transfromation call? 
    # If so, the output is expected to be of type TransformationResult.
    # If not, the output is expected to be of type pl.DataFrame.

    transform_name:Optional[str] = None # name of transformation
    transform_module:Optional[str] = None # module where this transformation belongs to
    transform_record: Optional[TransformationRecord] = None # Will only be Not none in blueprints.
    is_custom:bool = False 

    def get_args(self) -> str:
        return self.args
    
    def drop_args(self) -> Self:
        self.args = None
    
    def __str__(self) -> str:
        text = f"Function: {self.name} | Module: {self.module} | Arguments:\n{self.args}\n"
        text += f"Brief description: {self.desc}"
        if self.is_transform:
            text += "\nThis step is a transformation step."
        if self.is_custom:
            text += "\nThis step is a user defined function."
        return text

@dataclass
class ExecPlan():
    steps:list[ExecStep]

    def __iter__(self) -> Iterable[ExecStep]:
        return iter(self.steps)

    def __str__(self) -> str:
        if len(self.steps) > 0:
            text = ""
            for i, item in enumerate(self.steps):
                text += f"--- Step {i+1}: ---\n"
                text += str(item)
                text += "\n"
            return text
        else:
            return "No step has been set."
        
    def __len__(self) -> int:
        return len(self.steps)
    
    def clear(self) -> None:
        self.steps.clear()

    def is_empty(self) -> bool:
        return len(self.steps) == 0

    def add(self, step:ExecStep) -> None:
        self.steps.append(step)

    def popleft(self) -> Optional[ExecStep]:
        return self.steps.pop(0)

    def add_step(self
        , func:Callable[[pl.DataFrame, T], pl.DataFrame|TransformationResult]
        , desc:str
        , args:dict[str, Any] # Technically is not Any, but Anything that can be serialized by orjson..
        , is_transform:bool=False) -> None:
        
        self.steps.append(
            ExecStep(func.__name__, func.__module__, desc, args, is_transform, False)
        )

    def add_custom_step(self
        , func:Callable[[pl.DataFrame, T], pl.DataFrame|TransformationResult]
        , desc:str
        , args:dict[str, Any]
        , is_transform:bool=False) -> None:
        
        self.steps.append(
            ExecStep(func.__name__, func.__module__, desc, args, is_transform, True)
        )

def _select_cols(df:pl.DataFrame, cols:list[str]) -> pl.DataFrame:
    return df.select(cols)

def _lower_columns(df:pl.DataFrame) -> pl.DataFrame:
    return df.rename({c: c.lower() for c in df.columns})

class PipeBuilder:

    def __init__(self, project_name:str="my_project"):

        self.target:str = ""
        self.data:Optional[pl.DataFrame] = None
        self.project_name:str = project_name
        self._built:bool = False
        self._execution_plan:ExecPlan = ExecPlan(steps=[])
        self._blueprint:ExecPlan = ExecPlan(steps=[])
    
    def __len__(self) -> int: # positive int
        return len(self._execution_plan)
    
    def __str__(self) -> str:
        if not self._execution_plan.is_empty():
            text = f"Project name: {self.project_name}\nTotal steps: {len(self)} | Ready to build: {self._is_ready()} |"
            if self.target != "":
                text += f" Target variable: {self.target}"
            text += "\n"
            text += str(self._execution_plan)
            return text
        return "The current builder has no execution plan."
    
    ### I/O
    def set_target(self, target:str) -> Self:
        if target == "":
            raise ValueError("Target cannot be empty string.")
        
        if self.data is not None:
            if target not in self.data.columns:
                raise ValueError("Target is not found in the dataframe.")
            
        self.target = target 
        return self
        
    def set_data(self, df:pl.DataFrame|pd.DataFrame) -> Self:
        '''Set the data on which to "fit" the pipeline.'''
        try:
            if isinstance(df, pd.DataFrame):
                logger.warning("Found input to be a Pandas Dataframe. It will be converted to a Polars dataframe, "
                            "and the original Pandas dataframe will be erased.")
                
                self.data = pl.from_pandas(df)
                df = df.iloc[0:0]
            elif isinstance(df, pl.DataFrame):
                self.data = df.clone()
            else:
                raise TypeError("Input df must either be Pandas or Polars dataframe.")
        except Exception as e:
            logger.error(e)

        return Self
    
    def set_data_and_target(self, df:pl.DataFrame|pd.DataFrame, target:str) -> Self:
        _ = self.set_target(target)        
        _ = self.set_data(df)
        if target not in self.data.columns:
            raise ValueError("Target is not found in the dataframe.")
        
        self.target = target
        return self 

    def from_csv(self, path:str|Path, **csv_args) -> Self:
        '''
        
        '''
        try:
            self.data = pl.read_csv(path, **csv_args)
            return Self
        except Exception as e:
            logger.error(e)

    def from_parquet(self, path:str|Path, **parquet_args) -> Self:
        '''
        
        '''
        try:
            self.data = pl.read_parquet(path, **parquet_args)
            return Self
        except Exception as e:
            logger.error(e)
    
    def from_json(self, path:str|Path, **json_args) -> Self:
        '''
        
        '''
        try:
            self.data = pl.read_json(path, **json_args)
            return Self
        except Exception as e:
            logger.error(e)

    def from_excel(self, path:str|Path, **excel_args) -> Self:
        '''
        
        '''
        try:
            self.data = pl.read_json(path, **excel_args)
            return Self
        except Exception as e:
            logger.error(e)
    ### End of I/O
    
    ### Miscellaneous
    def show(self):
        print(self)

    def clear(self):
        self.data = self.data.clear()
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
    
    def set_lower_cols(self) -> Self:
        self._execution_plan.add_step(
            func = _lower_columns,
            desc = BuiltinExecutions.LOWER.value,
            args = {}
        )
        return self

    ### End of Miscellaneous

    ### Checks
    def _is_ready(self) -> bool:
        if self.data is None:
            return False
        
        return self.target != "" and self.target in self.data.columns
    
    ### End of Checks.

    ### Project meta data section
    def set_project_name(self, project_name:str) -> Self:
        if project_name == "":
            raise ValueError("Project name cannot be empty.")
        self.project_name = project_name
        return self

    ### End of project meta data section
    
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
        
        if self._is_ready():
            self._execution_plan.add_step(
                func = var_removal,
                desc = BuiltinExecutions.VAR_REMOVAL.value.format(threshold),
                args = {"threshold":threshold, "target":self.target},
            )
            return self
        else:
            raise ValueError("Target must be set before setting var removal.")
    
    def set_const_removal(self, include_null:bool=True) -> Self:
        
        self._execution_plan.add_step(
            func = constant_removal,
            desc = BuiltinExecutions.CONST_REMOVAL.value,
            args = {"include_null":include_null},
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
        '''Removes columns that satisfies the regex rule. May optionally only check on lowercased column names.'''
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
        '''Removes the given columns.'''
        self._execution_plan.add_step(
            func = remove_if_exists,
            desc = BuiltinExecutions.COL_REMOVAL.value,
            args = {"to_drop":cols},
        )
        return self
    
    def set_date_removal(self) -> Self:
        '''Removes columns that are inferred to be dates.'''
        self._execution_plan.add_step(
            func = date_removal ,
            desc = BuiltinExecutions.DATE_REMOVAL.value,
            args = {}
        )
        return self 
        
    ### End of column removal section

    ### Scaling and Imputation
    def set_scaling(self, cols:list[str]
        , strategy:ScalingStrategy=ScalingStrategy.NORMALIZE
        , const:int=1) -> Self:
        # const only matters if startegy is constant
        self._execution_plan.add_step(
            func = scale,
            desc = BuiltinExecutions.SCALE.value.format(strategy),
            args = {"cols":cols, "strategy": strategy, "const":const},
            is_transform = True
        )
        return self
    
    def set_impute(self, cols:list[str]
        , strategy:ImputationStartegy=ImputationStartegy.MEDIAN
        , const:int=1) -> Self:
        
        # const only matters if startegy is constant
        self._execution_plan.add_step(
            func = impute,
            desc = BuiltinExecutions.IMPUTE.value.format(strategy),
            args = {"cols":cols, "strategy": strategy, "const":const},
            is_transform = True
        )
        return self
    
    ### End of Scaling and Imputation

    ### Encoding
    def set_binary_encoding(self, cols:Optional[list[str]]=None) -> Self:
        
        if cols:
            description = BuiltinExecutions.BINARY_ENCODE.value
        else:
            description = "Automatically detect binary columns and turn them into [0,1] values by their order."

        self._execution_plan.add_step(
            func = binary_encode,
            desc = description,
            args = {"cols":cols},
            is_transform = True
        )
        return self

    def set_ordinal_encoding(self, mapping:dict[str, dict[str,int]], default:Optional[int]=None) -> Self:
        
        self._execution_plan.add_step(
            func = ordinal_encode,
            desc = BuiltinExecutions.ORDINAL_ENCODE.value,
            args = {"ordinal_mapping":mapping, "default": default},
            is_transform = True
        )
        return self
    
    def set_ordinal_auto_encoding(self, cols:list[str], default:Optional[int]=None) -> Self:
        
        self._execution_plan.add_step(
            func = ordinal_auto_encode,
            desc = BuiltinExecutions.ORDINAL_AUTO_ENCODE.value,
            args = {"cols":cols, "default": default},
            is_transform = True
        )
        return self
    
    def set_target_encoding(self, cols:list[str], min_samples_leaf:int=20, smoothing:int=10) -> Self:
        
        if self._is_ready():
            self._execution_plan.add_step(
                func = smooth_target_encode,
                desc = BuiltinExecutions.TARGET_ENCODE.value,
                args = {"target":self.target, "cols": cols, "min_samples_leaf":min_samples_leaf
                        , "smoothing":smoothing},
                is_transform = True
            )
            return self
        else:
            raise ValueError("The target must be set before target encoding.")
        
    def set_one_hot_encoding(self, cols:Optional[list[str]]=None, separator:str="_") -> Self:
        
        if cols:
            description = BuiltinExecutions.ORDINAL_AUTO_ENCODE.value
        else:
            description = "Automatically detect string columns and one-hot encode them."

        self._execution_plan.add_step(
            func = one_hot_encode,
            desc = description,
            args = {"cols":cols, "separator": separator},
            is_transform = True
        )
        return self
    
    def set_percentile_encoding(self, cols:list[str]) -> Self:

        self._execution_plan.add_step(
            func = percentile_encode,
            desc = BuiltinExecutions.PERCENTILE_ENCODE.value,
            args = {"cols":cols},
            is_transform = True
        )
        return self
    
    ### End of Encoding Section

    ### Custom Actions

    # Step and transforms are differentiated in order to prevent error.
    # If the custom action/transformation of func is not dependent on the dataset (self.data)
    # then you should use add_custom_step.
    #
    # If the transformation is dependent on information about the dataset (self.data),
    # then the output type of func should be something that inherits from TransformationResult.
    # We need a "record" that holds enough information about (self.data) to fully repeat
    # this transformation.


    def add_custom_step(self
        , func:Callable[Concatenate[pl.DataFrame, P], pl.DataFrame]
        , desc:str
        , args:dict[str, Any]
        ) -> Self:

        if "return" not in func.__annotations__:
            logger.info("It is highly recommended that the custom step returns " 
                        "a Polars dataframe. If this returns a TransformationResult, use add_custom_transform instead.")

        self._execution_plan.add_custom_step(
            func = func,
            desc = desc,
            args = args,
            is_transform = False
        )

        return self
    # Should I rename transform to something else?    
    def add_custom_transform(self
        , func:Callable[Concatenate[pl.DataFrame, P], TransformationResult]
        , desc:str
        , args:dict[str, Any]
        ) -> Self:

        if "return" not in func.__annotations__:
            logger.info("It is highly recommended that the custom transform returns "
                        "a type that inherits from TransformationResult."
                        "If this returns a pl.DataFrame, use add_custom_step instead.")

        self._execution_plan.add_custom_step(
            func = func,
            desc = desc,
            args = args,
            is_transform = True
        )

        return self

    ### End of Custom Actions

    def fit(self, X, y) -> Self:
        pass

    def build(self
              , df:Optional[pl.DataFrame|pd.DataFrame]=None
              ) -> pl.DataFrame:
        '''
            Build according to the steps.

            Arguments:
                df: Another chance to set input df if it has not been set.

            Returns:
                A dataframe.
        
        '''
        if df is not None:
            _ = self.set_data(df)
        
        n = len(self._execution_plan)
        logger.info(f"Starting to build. Total steps: {n}.")
        if not self._is_ready():
            raise ValueError(f"Dataframe is not set properly, or the target {self.target} is not found "
                             "in the dataframe.")
        
        if self._built:
            logger.warning("The PipeBuilder is built once already. It is not intended to be built again. "
                           "To avoid unexpected behavior, construct a new pipe only after calling .clear().")
        
        i = 0
        # Todo! If something failed, save a backup dataframe to a temp folder.
        while not self._execution_plan.is_empty():
            i += 1
            step = self._execution_plan.popleft()
            logger.info(f"|{i}/{n}|: Executed Step: {step.name} | is_transform: {step.is_transform}")
            start = perf_counter()
            success = True
            if step.is_transform: # Essentially the fit step.
                apply_transf:Callable[Concatenate[pl.DataFrame, P], TransformationResult] 
                apply_transf = getattr(importlib.import_module(step.module), step.name)
                rec: TransformationRecord
                self.data, rec = apply_transf(self.data, **step.args)
                new_step = ExecStep(
                    name = step.name,
                    module = "N/A",
                    desc = step.desc, # 
                    # args = step.args, # don't need this when applying
                    is_transform = True,
                    transform_name = type(rec).__name__,
                    transform_module = rec.__module__,
                    transform_record = rec,
                    is_custom = step.is_custom
                )
                self._blueprint.add(new_step)
            else:
                apply_func:Callable[Concatenate[pl.DataFrame, P], pl.DataFrame]  
                apply_func = getattr(importlib.import_module(step.module), step.name)
                
                self.data = self.data.pipe(apply_func, **step.args)
                self._blueprint.add(step)

            end = perf_counter()
            logger.info(f"|{i}/{n}|: Finished in {end-start:.2f}s | Success: {success}")

        logger.info("Build success. A blueprint has been built and can be viewed by calling .blueprint(), "
                    "and can be saved as a json by calling .write()")

        self._built = True
        output = self.data.clone()
        return output
    
    def get_final_data(self) -> Optional[pl.DataFrame]:
        if self._built:
            return self.data.clone()
        else:
            logger.info("Pipeline has not been built. There cannot be any final data.")
            return None
    
    # Rename this in the future?
    def apply(self, df:pl.DataFrame|pd.DataFrame) -> pl.DataFrame:
        if not self._built:
            raise ValueError("The builder must be built before applying it to new datasets.")
        
        if isinstance(df, pd.DataFrame):
            logger.warning("Found input to be a Pandas dataframe. Turning it into a Polars dataframe.")
            try:
                input_df:pl.DataFrame = pl.from_pandas(df)
            except Exception as e:
                logger.error(e)
        else:
            input_df:pl.DataFrame = df

        n = len(self._blueprint)
        step:ExecStep
        for i, step in enumerate(self._blueprint):
            logger.info(f"|{i+1}/{n}|: Performing Step: {step.name} | is_transform: {step.is_transform}")
            start = perf_counter()
            success = True
            if step.is_transform:
                try:
                    rec:TransformationRecord = step.transform_record
                    input_df = input_df.pipe(rec.transform)
                except Exception as e:
                    success = False
                    logger.error(e)
            else:
                apply_func:Callable[Concatenate[pl.DataFrame, P], pl.DataFrame] 
                apply_func = getattr(importlib.import_module(step.module), step.name)
                input_df = input_df.pipe(apply_func, **step.args)

            end = perf_counter()
            logger.info(f"|{i+1}/{n}|: Finished in {end-start:.2f}s | Success: {success}")
        
        return input_df

    def blueprint(self):
        return print(self._blueprint)
        
    def write(self, name:str="") -> None:
        if self._blueprint.is_empty():
            logger.warning("Blueprint is empty. Nothing is done.")
            return

        directory = "./blueprints/"
        if name == "":
            name += self.project_name + ".json"
            logger.info(f"No name is specified, using project name ({name}) as default.")
        else:
            if not name.endswith(".json"):
                name += ".json"

        if not os.path.isdir(directory):
            logger.info("Local ./blueprints/ directory is not found. It will be created.")
            os.mkdir(directory)
        try:
            data = orjson.dumps(self._blueprint, option=orjson.OPT_NON_STR_KEYS|orjson.OPT_SERIALIZE_NUMPY)
            destination = directory+name
            with open(destination, "wb") as f:
                f.write(data)

            logger.info(f"Successfully saved to {destination}.")
        except Exception as e:
            logger.error(e)

    def from_blueprint(self, path:str|Path):
        logger.info("Reading from a blueprint. The builder will reset itself.")
        self.clear()
        try:
            f = open(path, "rb")
            data = orjson.loads(f.read())
            f.close()
            steps:list[dict[str, Any]] = data["steps"]
            for s in steps:
                if s["is_transform"]: # Need to recreate TransformRecord objects from dict
                    name = s.get("transform_name", None)
                    module = s.get("transform_module", None)
                    record = s.get("transform_record", None)
                    if (name is not None) or (module is not None) or (record is not None):
                        if (name is None) or (module is None) or (record is None):
                            raise ValueError(f"Something went wrong with the transform: {s['name']}. "
                                             "All transform_name, transform_module and transform_record fields "
                                             "must not be null.")
                    # Get the class the TransformationRecord belongs to.
                    c = getattr(importlib.import_module(module), name)
                    # Create an instance of c
                    rec = c(**record)
                    s["transform_record"] = rec # Make transform_record to be a real TransformRecord object.

                self._blueprint.add(ExecStep(**s))

            self._built = True
            logger.info("Successfully read from a blueprint.")
        except Exception as e:
            logger.error(e)

    ### End of IO Section 



