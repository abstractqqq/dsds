# Welcome to the DSDS

This package is in pre-alpha stage. There is currently no documentation webpage. Only docstrings and github. Please read [CONTRIBUTING.md](./CONTRIBUTING.md) if you are a developer interested in contributing to this package.

!!! Restructure Warning: Polars Extension Submodule is going to be an independent package and is the main focus of development right now. See [here](https://github.com/abstractqqq/polars_ds_extension) !!!

A general purpose traditional data science package for large and scalable data operations. It aims to be an improvement over a subset of functionalities in other packages like Scikit-learn, category_encoders, or feature engine, etc. The primary focus right now is:

0. Treating dataframes as first-class citizen, using Rust, NumPy to support. The only other core dependency is SciPy and NumPy. Will rely more on Rust as time goes. Python side dependency should be minimized.
1. Providing practical feature prescreen (immediate detection and removal of useless featuers, data profiling, over time report etc.)
2. Fast and furious feature selection and engineering using simple methods. It has significantly faster F-score, mutual_info_score, better feature extraction and engineering, etc. Currently, DSDS provides MRMR with more than 90% speed up than the most popular MRMR package on github. For more details, see [here](./FEATURE_SELECTION.md)
3. Cleaner pipeline construction and management, much easier to create custom "transformers" (All Polars Expressions are acceptable as "transformers". The current blueprint module probably needs an overhaul)
4. Consistent API with intuitive argument names and extensive docstring and examples. 
5. Functional interface and fully typed functions for a better developer experience. No mixins, no multiple inheritance. No classes. Just Polars.
6. Better performance and faster iteration

At this moment, it will not include traditional data science models, like SVM, random forest, etc. This may change once in the future when the Rust side of data science models catch up.

DSDS is built upon your favorite: [Polars Dataframe](https://github.com/pola-rs/polars)

[Here is why in my opinion data scientists, analysts and engineers should switch to Polars!](https://medium.com/@tq9695/all-that-polars-that-make-you-forget-pandas-3dc0fdfaefbe)

If you are interested in Polars powered Time Series Feature Extraction, please check out [functime](https://github.com/neocortexdb/functime) (I am also a contributor to that project)

## Usage

Practical Feature Prescreen and Data 
```python
from dsds.prescreen import (
    infer_highly_correlated,
    infer_str_cols,
    infer_by_var,
    infer_invalid_numeric,
    infer_conti,
    infer_emails,
    infer_nulls,
    infer_by_pattern,
    over_time_report_num,
) # and a lot more!
```
![dd](./pics/dependency_detection.PNG)


Functional Pipeline Interface which supports both Eager and LazyFrames! And it can be pickled and load back and reapplied! See more in the [examples on github.](./examples/pipeline.ipynb)

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

Common practical Transformations for feature extraction or engineering:
```Python
import dsds.transform as t
df = pl.DataFrame({
    "survey":["0% of my time", "1% to 25% of my time", "75% to 99% of my time", 
            "50% to 74% of my time", "75% to 99% of my time", 
          "50% to 74% of my time"]
})
print(t.extract_patterns(df, "(\d*\.?\d+)%", ["survey"], join_by=" to "))

shape: (6, 1)
┌────────────┐
│ survey     │
│ ---        │
│ str        │
╞════════════╡
│ 0%         │
│ 1% to 25%  │
│ 75% to 99% │
│ 50% to 74% │
│ 75% to 99% │
│ 50% to 74% │
└────────────┘
#-----------------------------------------------------------------------------
import dsds.transform as t
df = pl.DataFrame({
    "a":[1, 2, 3],
    "b":[1, 2, 3],
    "c":[1, 2, 3]
})
t.extract_horizontally(df, cols=["a", "b", "c"], extract=["min", "max", "sum"])

shape: (3, 3)
┌────────────┬────────────┬────────────┐
│ min(a,b,c) ┆ max(a,b,c) ┆ sum(a,b,c) │
│ ---        ┆ ---        ┆ ---        │
│ i64        ┆ i64        ┆ i64        │
╞════════════╪════════════╪════════════╡
│ 1          ┆ 1          ┆ 3          │
│ 2          ┆ 2          ┆ 6          │
│ 3          ┆ 3          ┆ 9          │
└────────────┴────────────┴────────────┘
#-----------------------------------------------------------------------------
import dsds.transform as t
df = pl.DataFrame({
    "date1":["2021-01-01", "2022-02-03", "2023-11-23"]
    , "date2":["2021-01-01", "2022-02-03", "2023-11-23"]
}).with_columns(
    pl.col(c).str.to_date() for c in ["date1", "date2"]
)
print(df)

shape: (3, 2)
┌────────────┬────────────┐
│ date1      ┆ date2      │
│ ---        ┆ ---        │
│ date       ┆ date       │
╞════════════╪════════════╡
│ 2021-01-01 ┆ 2021-01-01 │
│ 2022-02-03 ┆ 2022-02-03 │
│ 2023-11-23 ┆ 2023-11-23 │
└────────────┴────────────┘

cols = ["date1", "date2"]
print(t.extract_dt_features(df, cols=cols))

shape: (3, 8)
┌────────────┬────────────┬────────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│ date1      ┆ date2      ┆ date1_year ┆ date2_yea ┆ date1_qua ┆ date2_qua ┆ date1_mon ┆ date2_mon │
│ ---        ┆ ---        ┆ ---        ┆ r         ┆ rter      ┆ rter      ┆ th        ┆ th        │
│ date       ┆ date       ┆ u32        ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       │
│            ┆            ┆            ┆ u32       ┆ u32       ┆ u32       ┆ u32       ┆ u32       │
╞════════════╪════════════╪════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╡
│ 2021-01-01 ┆ 2021-01-01 ┆ 2021       ┆ 2021      ┆ 1         ┆ 1         ┆ 1         ┆ 1         │
│ 2022-02-03 ┆ 2022-02-03 ┆ 2022       ┆ 2022      ┆ 1         ┆ 1         ┆ 2         ┆ 2         │
│ 2023-11-23 ┆ 2023-11-23 ┆ 2023       ┆ 2023      ┆ 4         ┆ 4         ┆ 11        ┆ 11        │
└────────────┴────────────┴────────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
```

Benchmarks (All done on Linux):

![Screenshot](./pics/bench.png)

DSDS currently provides Polars plugins but the API is subject to change.

![Plugin](./pics/plugin.PNG)


## Dependencies

Python 3.9, 3.10, 3.11+ is recommended.

It should run on all versions >= 3.9.


# Why the name DSDS?

Originally I choose the name DSDS because it stands for Dark Side of Data Science. I thought it is kind of cool and since I am doing things differently, it justifies "dark side". I think data science/modelling infrastructure is the most foundational work that is also the most under-appreciated. Pipelines are buried "in darkness" most of the time. Feature selection is often considered a dark art, too. So the name DarkSide/dsds really makes sense to me. But right now, DSDS is just a name. I don't know think calling it Dark Side of Data Science helps with recoginition and I personally wouldn't call the way I am doing things the "dark side" any more. It is simply a new way of doing things not explored by people before.

# Why is this package dependent on Sklearn?

As of 0.0.34, the package does not explicitly rely on Scikit-learn any more. The only model-dependent "feature_importance" the package will use is LightGBM, which is an optional dependency.

# Why not write more in Rust?

Yes. I am. I recently resumed working on traditional NLP, and this is an area where Rust really shines. It is well-known that Stemmers in NLTK has terrible performance, but stemming is an important operations for many traditional NLP algorithms. To combat this, I have decided to move stemming and other heavy string manipulations to Rust and leave only a very thin wrapper in Python. That said, using Rust can greatly improve performance in many other modules, not just dsds.text. But because development in Rust is relatively slow, I do not want to blindly 'rewrite' in Rust.

# Contribution

See CONTRIBUTING.md for my contact info.

# Disclaimer

Snowball Stemming implementation for Rust is taken from Tsoding's Seroost project.