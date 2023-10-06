# Welcome to the DSDS

This package is in pre-alpha stage. Please read CONTRIBUTING.md if you are a developer interested in contributing to this package.

A general purpose traditional data science package for large and scalable data operations. It aims to be an improvement over a subset of functionalities in other packages like Scikit-learn, category_encoders, or feature engine, etc. The primary focus right now is:

1. Providing practical feature prescreen (immediate detection and removal of useless featuers, data profiling, over time report etc.)
2. Fast and furious feature selection and engineering using simple methods. It has significantly faster F-score, MRMR, mutual_info_score, better feature extraction APIs, etc.
3. Cleaner pipeline construction and management, much easier to create custom "transformers" (All Polars Expressions under the hood)
4. Consistent API with intuitive argument names and extensive docstring and examples for not-easy-to-explain stuff.
5. Functional interface and fully typed functions for a better developer experience. No mixins, no multiple inheritance. No classes. Just Polars.
6. Better performance and faster iteration and less reliance on expensive VMs

At this moment, it will not include traditional data science models, like SVM, random forest, etc. This may change once in the future when the Rust side of data science models catch up.

DSDS is built around your favorite: [Polars Dataframe](https://github.com/pola-rs/polars)

[Here is why in my opinion data scientists, analysts and engineers should switch to Polars!](https://medium.com/@tq9695/all-that-polars-that-make-you-forget-pandas-3dc0fdfaefbe)

## Usage

Practical Feature Prescreen
```python
from dsds.prescreen import (
    drop_if_exists,
    drop_by_pattern,
    drop_by_regex,
    drop_by_var,
    drop_highly_unique,
    drop_constants,
    drop_dates,
    drop_highly_null,
    infer_invalid_numeric,
    infer_conti,
    infer_emails,
    infer_nulls,
) # and a lot more!

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

Benchmarks: (using a version not available to the public yet.)

![Screenshot](./pics/benches.PNG)

DSDS currently provides Polars plugins but the API is subject to change.

![Plugin](./pics/plugin.PNG)


## Dependencies

Python 3.9, 3.10, 3.11+ is recommended.

It should run on all versions >= 3.9.

Note: scikit-learn, lightgbm are needed for full functionalities. I am writing a lot of things from scratch in order to minimize dependency and increase efficiency within the package. 


# Why the name DSDS?

Originally I choose the name DSDS because it stands for Dark Side of Data Science. I thought it is kind of cool and since I am doing things differently, it justifies "dark side". I think data science/modelling infrastructure is the most foundational work that is also the most under-appreciated. Feature selection is often considered a dark art, too. So the name DarkSide/dsds really makes sense to me.

# Why is this package dependent on Sklearn?

You are right in the sense that this package does its best to separate itself from sklearn because of its focus and design. You do not need sklearn for pipelines, transformations, metrics, or the prescreen modules. However, for the fs (feature selection) module, right now there is no other high quality, tried and true package for some models. The feature importance from these two models are used in some feature selection algorithms. Feel free to let me know if there are alternatives.

# Why not write more in Rust?

Yes. I am. I recently resumed working on traditional NLP, and this is an area where Rust really shines. It is well-known that Stemmers in NLTK has terrible performance, but stemming is an important operations for many traditional NLP algorithms. To combat this, I have decided to move stemming and other heavy string manipulations to Rust and leave only a very thin wrapper in Python. That said, using Rust can greatly improve performance in many other modules, not just dsds.text. But because development in Rust is relatively slow, I do not want to blindly 'rewrite' in Rust. In near future, I do not have plans to 'rewrite' in Rust, if performance gain is less than 10%.

# Contribution

See CONTRIBUTING.md for my contact info.

# Disclaimer

Snowball Stemming implementation for Rust is taken from Tsoding's Seroost project.