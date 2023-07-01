# Welcome to the Dark Side of Data Science (DSDS)

This package is in pre-alpha stage. Please read CONTRIBUTING.md if you are a developer interested in contributing to this package.

Welcome to DSDS, a data science package that aims to be an improvement over a subset of sklearn's functionality, primarily in the following areas:

1. Providing practical feature prescreen (immediate detection and removal of useless featuers, data profiling, etc.)
2. Fast and furious feature selection (significantly faster F-score, MRMR, mutual_info_score, etc.)
3. Cleaner pipeline construction and management (See examples below.)
4. Compatible with Polars LazyFrames (Yes! Pipelines can enjoy the benefits of query optimization too!)
5. Functional interface and fully typed functions for a better developer experience. No mixins, no multiple inheritance. No classes. No nonsenses.

DSDS is built around your favorite: [Polars Dataframe](https://github.com/pola-rs/polars)

## Usage

Practical Feature Prescreen
```python
from dsds.prescreen import (
    remove_if_exists
    , regex_removal
    , pattern_removal
    , var_removal
    , null_removal
    , unique_removal
    , constant_removal
    , date_removal
    , non_numeric_removal
    , get_unique_count
    , get_string_cols
    , suggest_normal
    , discrete_inferral
)


```

Functional Pipeline Interface which supports both Eager and LazyFrames! And it can be pickled and load back and reapplied! See more in the examples on github.

```python
from dsds.prescreen import *
from dsds.transform import *

input_df = df.lazy()
output = input_df.pipe(var_removal, threshold = 0.5, target = "Clicked on Ad")\
    .pipe(binary_encode)\
    .pipe(ordinal_auto_encode, cols = ["City", "Country"])\
    .pipe(impute, cols=["Daily Internet Usage", "Daily Internet Usage Band", "Area Income Band"], strategy="median")\
    .pipe(impute, cols=["Area Income"], strategy = "mean")\
    .pipe(scale, cols=["Area Income", "Daily Internet Usage"])\
    .pipe(one_hot_encode, cols= ["One_Hot_Test"])\
    .pipe(remove_if_exists, cols = ["Ad Topic Line", "Timestamp"])\
    .pipe(mutual_info_selector, target = "Clicked on Ad", top_k = 12)
```

Performance without sacrificing user experience. See ./examples/dsds_comparisons.ipynb

![Screenshot](./pics/impute.PNG)

## Dependencies

Python 3.9, 3.10, 3.11+ is recommended. We are forward looking.

pip install polars scipy numpy

pip install dsds[all]

Note: scikit-learn, lightgbm, xgboost are needed for full functionalities. 

# Why DSDS?

I choose DarkSide because data pipelines are like real life pipelines, buried under the ground. It is the most foundational work that is also the most under-appreciated component of any data science project. Feature selection is often considered a dark art, too. So the name DarkSide/dsds really makes sense to me.