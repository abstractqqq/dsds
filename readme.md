# Welcome to the Dark Side of Data Science

Still in early development. Name of package undecided yet, well, probably it will be called the DarkSide or DSDS (Dark Side for Data Science). I choose DarkSide because data pipelines are like real life pipelines, buried under the ground. It is the most foundational work that is also the most under-appreciated component of any data science project. Feature selection is often considered a dark art, too. So the name DarkSide/dsds really makes sense to me.

This library aims to be a lightweight altenative to Scikit-learn (Sklearn), especially in the data preparation stage, e.g. feature screening/selection, basic transformations (scale, impute, one-hot encode, target encode, etc.) Its goal is to replace sklearn's pipeline. (Everything except the models are rewritten. The current dataset builder does not have model steps yet.). Its focuses are on:

1. Being more dataframe centric in design. Dataframe in, dataframe out, and try not to convert or copy to NumPy unless necessary, or provide low-memory options.

2. Performance. Most algorithms are rewritten and are 3-5x faster, 10x if you have more cores on your computer, than Scikit-learn's implementation.

3. Simplicity and consistency. This library should not be everything. It should stick to the responsibilities outlined above. It shouldn't become a visualization library. It shouldn't overload users with millions of input options, most of which won't be used anyway and which really adds little but side effects to the program. It shouldn't be a package with models. (We might add some wrappers to Scipy for EDA). This package helps you build and manage the pipeline, from feature selection to basic transformations, and provides you with a powerful builder to build your pipe!

4. Provide more visibility into data pipelines without all the pomp of a web UI. Make data pipelines editable outside Python.

5. Be more developer friendly by introducing useful types and data structures in the backend.

To this end, I believe the old "stack", Pandas + Sklearn + some NumPy, is inadequate, mostly because

1. Their lack of parallelism
2. Pandas's "object" types making things difficult and its slow performance.
3. Lack of types enforcement, leading to infinitely many quality checks. Lack of types describing outputs.

Dask and PySpark are distributed systems and so are their own universe. But on a single machine, Polars has proven to be more performant and less memory intensive than both of them.

Most algorithms in Sklearn are available in Scipy, and Scipy relies more heavily on C and usually has multicore options. Therefore, when the algorithm is too complex to perfom in Polars, we can rely on Scipy. 

So the proposed new "stack" is Polars + Scipy + some NumPy.

Note, if you want everything to work, you may need to install sklearn because in some algorithms we need random forest's feature importances.

# Existing Functionalities:

## EDA Prescreen

The point of feature prescreening is to reduce the number of feature to be analyzed by dropping obviously useless features. E.g in a dataset with 500+ features, it is impossible for a person to know what the features are. It will take very long time to run feature selection on all features. In this case we can quickly remove all features that are constant, all id column feautres, or all features that are too unique (which makes them like ids). If you are confident in removing columns with high null percentage, you may do that do.

1. Data profiling for an overview of the statistics of the data.

2. Infer/remove columns based on column data type, null pct, unique pct, variance, constant, or name of column.

### todo()!

1. Infer duplicate columns, string columns hiding as dates, distribution of data in column.

2. Remove based on the above (less trivial) characteristics.

## EDA Transformation

1. Binary transforms, boolean transform, ordinal encoding, auto ordinal encoding, one-hot encoding, target encoding.

2. Imputation and scaling.

### todo()!

1. More imputation and scaling startegies.

2. More slightly advanced encoding techniques.

## EDA Selection

Feature selection done fast. May need more optimization.

0. Methods based on entropy: mutual_info, naive_sample_ig
    
    Let X denote the test column/feature, and Y the target. We compute the conditional entropy H(Y|X), which represents the remaining randomness in the random variable Y, given the random variable X. For more details, see [here](https://en.wikipedia.org/wiki/Entropy_(information_theory)).

    The mutual_info function is a speed-up version of sklearn's mutual_info_classif, but with some small precision issues right now. (Not sure if this is a bug or not. Most likely this is just a precision issue. Need some help.)

    More details can be found in the docstring of the functions.

1. Classic Anova One Way F-test.
    
    See [here](https://saylordotorg.github.io/text_introductory-statistics/s15-04-f-tests-in-one-way-anova.html).

2. Basic MRMR Algorithm with many variations: mrmr, knock-out-mrmr.

    See [here](https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b)

## EDA Builder

A builder that helps you with the data preparation part of the ML cycle. Its aim is to create blueprints, reusable formula for recreating the same pipeline and should be editable without code. It is essentially like Sklearn's pipeline, but less object dependent and easier to serialize and edit.

1. Connect selections into builder, thus incorporating feature selection into the pipeline.

2. Enable logging so that it actually writes to a log file.

## EDA Misc

Miscallenous functions.

## EDA Text (Halted.)

1. A quick reusable method to transform text data in a column into multiple numerical columns. Great for TF-IDF kind of thing and also any other model that cannot consume text data directly. Experimental for now. Not suitable for big data. 

## Dependencies

Python 3.11+ is recommended. We are forward looking.

pip install polars orjson scipy numpy

Note: nltk, scikit-learn, and xgboost are needed for full functionalities. 

(nltk may require additional downloads.)

## General Considerations and Guidelines Before Making Contributions:

0. All guidelines below can be discussed and are merely guidelines which may be challenged.

1. If you can read the data into memory, your code should process it faster than Scikit-learn and Pandas. "Abuse" Polars' multi-core capabilities as much as possible before sending data to NumPy.

2. Provide proof that the algorithm generates exact/very close results to Scikit-learn's implementation.

3. Try not to include other core packages. NumPy, Scipy and Polars should be all. The preferred serialization strategy is dataclasses + Orjson, not pickling. Avoid nested dataclasses.

4. Fucntion annotaions are required and functions should have one output type only.

5. Obscure algorithms that do not have a lot of usages should not be included in the package. The package is designed in such a way that it can be customized (A lot more work to be done here.)