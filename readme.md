# Welcome to the Dark Side of Data Science (DSDS)

This package is in pre-alpha stage. Please read CONTRIBUTING.md if you are a developer interested in contributing to this package.

# Basic Examples

1.


# Overview of Existing Functionalities:

## prescreen

The point of feature prescreening is to reduce the number of feature to be analyzed by dropping obviously useless features. E.g in a dataset with 500+ features, it is impossible for a person to know what the features are. It will take very long time to run feature selection on all features. In this case we can quickly remove all features that are constant, all id column feautres, or all features that are too unique (which makes them like ids). If you are confident in removing columns with high null percentage, you may do that do.

1. Data profiling for an overview of the statistics of the data.

2. Infer/remove columns based on column data type, null pct, unique pct, variance, constant, or name of column.

### todo()!

1. Infer duplicate columns, string columns hiding as dates, distribution of data in column.

2. Remove based on the above (less trivial) characteristics.

## transform

1. Binary transforms, boolean transform, ordinal encoding, auto ordinal encoding, one-hot encoding, target encoding.

2. Imputation and scaling.

3. Power transform.

### todo()!

1. More imputation and scaling startegies.

2. More slightly advanced encoding techniques.

## Feature Selection (fs)

Feature selection done fast. May need more optimization.

0. Methods based on entropy: mutual_info, naive_sample_ig
    
    Let X denote the test column/feature, and Y the target. We compute the conditional entropy H(Y|X), which represents the remaining randomness in the random variable Y, given the random variable X. For more details, see [here](https://en.wikipedia.org/wiki/Entropy_(information_theory)).

    The mutual_info function is a speed-up version of sklearn's mutual_info_classif, but with some small precision issues right now. (Not sure if this is a bug or not. Most likely this is just a precision issue. Need some help.)

    More details can be found in the docstring of the functions.

1. Classic Anova One Way F-test.
    
    See [here](https://saylordotorg.github.io/text_introductory-statistics/s15-04-f-tests-in-one-way-anova.html).

2. Basic MRMR Algorithm with many variations: mrmr, knock-out-mrmr.

    See [here](https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b)

    Also see mrmr_examples.ipynb in the examples folder.

### todo()!

1. More known feature selection algorithms.

## Compatible with df.pipe()

A lot of work is being done. It is hard to explain everything. But right now this section is not good for eager dataframes.

Most functions in this package are compatible with lazy Polar's pipe operation, making the data preparation part of the ML cycle trivial. In addition, we can create blueprints, reusable formula for recreating the same pipeline. It is like Sklearn's pipeline, but you don't need to initialize any objects! Serialization is builtin in Polars with the write.json() function on lazy frames. The result is a much cleaner pipeline which follows "chain of thought". It is lazy Polars's execution plan under the hood. To get a blueprint out as in the builder example, please start with a lazy df as input.

### todo()!

1. Write more functions compatible with Polar's lazy execution plan. Make this pipeline more general purpose.

2. Enable logging so that it actually writes to a log file??

3. Maybe create another data structure to better manage the pipe?

4. Debugging and the precision issue.

### IMPORTANT: If you are reusing an execution plan, there is a known precision issue right now. However, the precision error is on the magnitude of 1e-8.

### IMPORTANT: It is possible to add your custom transformations into the pipeline. If the transformation is independent of incoming data, then no worries. If the transformation is dependent on the incoming data, then you have to write to in such a way that it will be "memorized" by the execution plan. Otherwise, the operation will not persist.

## Utils

Miscallenous functions.

## EDA Text (Halted.)

## Dependencies

Python 3.9, 3.10, 3.11+ is recommended. We are forward looking.

pip install polars scipy numpy

pip install dsds[all]

Note: scikit-learn, lightgbm, xgboost are needed for full functionalities. 

# Why DSDS?

I choose DarkSide because data pipelines are like real life pipelines, buried under the ground. It is the most foundational work that is also the most under-appreciated component of any data science project. Feature selection is often considered a dark art, too. So the name DarkSide/dsds really makes sense to me.