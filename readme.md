# Goals of this Data Analysis Toolkit

Still in early development. Name of package undecided yet.

Goal 1: Make traditional EDA, feature selection methods (especially those in sklearn) easier to perform, faster to compute, and more "DataFrame-friendly", meaning that 

1. Inputs should be dataframes and reduce copying to NumPy array as much as possible. This means we avoid internally copying data to numpy array.
2. Output should be clean dataframes on which we can quickly sort and filter.

To this end, we have to entirely use Polars (maybe make some methods pandas compatible in the future). The main reason for not using pandas is performance, ease of programming in polars, and pandas's loose type enforcement.

Goal 2: Remove dependency on sklearn for the data preparation part of the model building pipeline. ONLY the data preparation part for now.

Why? Here are some reasons: 

(a). In-memory objects are not the way to build long lasting pipelines because they are difficult to change/update, and use more resource than necessary, and are hard to serialize. 

(b). A lot of sklearn data preprocessing/feature selection algorithms suffer from immense performance bottlenecks, and are not "dataframe-friendly"!

# Existing Functionalities:

## EDA Prescreen

The point of feature prescreening is to reduce the number of feature to be analyzed by dropping obviously useless features. E.g in a dataset with 500+ features, it is impossible for a person to know what the features are. It will take very long time to run feature selection on all features. In this case we can quickly remove all features that are constant, all id column feautres, or all features that are too unique (which makes them like ids). If you are confident in removing columns with high null percentage, you may do that do.

0. Data profiling for an overview of the statistics of the data.

1. Infer/remove columns based on null pct, unique pct, variance, constant, or name of column.


## EDA Transformation

0. Binary transforms, boolean transform, ordinal encoding, auto ordinal encoding, one-hot encoding.

1. More advanced encoding techniques such as target encoding. More to come.

2. Imputation and scaling.

## EDA Selection

Feature selection done in a simple, fast, and "less memory intensive way". May need more optimization.

0. Methods based on entropy: mutual_info, naive_sample_ig
    
    Let X denote the test column/feature, and Y the target. We compute the conditional entropy H(Y|X), which represents the remaining randomness in the random variable Y, given the random variable X. For more details, see [here](https://en.wikipedia.org/wiki/Entropy_(information_theory)).

    The mutual_info function is a speed-up version of sklearn's mutual_info_classif. 

    More details can be found in the docstring of the functions.

1. Classic Anova One Way F-test.
    
    See [here](https://saylordotorg.github.io/text_introductory-statistics/s15-04-f-tests-in-one-way-anova.html).

2. Basic MRMR Algorithm:

    See [here](https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b)

## EDA Builder

A builder that helps you with the data preparation part of the ML cycle. Its aim is to create blueprints, reusable formula for recreating the same pipeline and should be editable without code. 

## EDA Misc

Miscallenous functions.

## EDA Text

1. A quick reusable method to transform text data in a column into multiple numerical columns. Great for TF-IDF kind of thing and also any other model that cannot consume text data directly. Experimental for now. Not suitable for big data. 

## Dependencies

Python 3.11+ is recommended. We are forward looking.

pip install polars orjson scipy nltk scikit-learn xgboost 

(nltk may require additional downloads.)

# todo()!

EDA Selection:

1. More feature selection algorithms that are not bloated and yield good results.

EDA Transformation:

1. WOE Transformation.

2. More imputation and scaling strategies.

EDA Builder:

1. Finish the main functionalities.

2. Checkpoint functionalities.

3. I/O from databases, S3, datalake, etc.

4. Allow user defined function calls into the pipeline.

EDA Text: Currently on hold.

EDA Prescreen:

1. Open to suggestions.

