# My Data Analysis Toolkit

Goal 1: Make traditional EDA, feature selection methods (especially those in sklearn) easier to perform, faster to compute, and more "DataFrame-friendly", meaning that 

1. Inputs should be dataframes and reduce copying to NumPy array as much as possible. This means we avoid internally copying data to numpy array.
2. Output should be clean dataframes on which we can quickly sort and filter.

To this end, we have to entirely use Polars as Pandas is not a great performance-wise and it has inconsistent data type issues. 

Goal 2: Create memory efficient way of scaling, transforming data. Speed may be compromised in the interest of low memory, but will be case dependent.

Goal 3: Remove dependency on sklearn for the data preparation part of the model building pipeline. ONLY the data preparation part. The models in sklearn are great.

Why? Here are some reasons: 
(a). In-memory objects are not the way to build long lasting pipelines because they are difficulty to change/update, and use more resource than necessary. 
(b). Old pandas + sklearn + NumPy algorithms pales in comparison with polars + minimal numpy in terms of performance, especially when we want to be "dataframe centric"!
(c). Easier to debug. Let's face the truth and speak the truth! It's hard to debug in OOP! I am working on some kind of a "lazy" builder that will combine feature removal, imputation, scaling,  

Goal 4: Dataframe-friendly text transformations, good for small scale NLP analysis. May not be efficient over large datasets.


## EDA Utils

My goal is to develop all these test and make them work on a Polars dataframe. A consistent API should be used. These are the methods I have so far.

0. Feature removal methods (null_pct, var, etc.) and binary transforms. Imputation and scaling will be added in teh future.

1. A method based on entropy. (Univariate as of now, but can be potentially multi-variate.)
    
    Let X denote the test column/feature, and Y the target. We compute the conditional entropy H(Y|X), which represents the remaining randomness in the random variable Y, given the random variable X. For more details, see [here](https://en.wikipedia.org/wiki/Entropy_(information_theory)).

    Also see the docstring of information_gain.

2. Classic Anova One Way F-test. (Univariate) 
    
    See [here](https://saylordotorg.github.io/text_introductory-statistics/s15-04-f-tests-in-one-way-anova.html).

3. MRMR Algorithm:

    See [here](https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b)

## Text data transformation (text_data.py)

1. A quick reusable method to transform text data in a column into multiple numerical columns. Great for TF-IDF kind of thing and also any other model that cannot consume text data directly. 

## Dependencies

Python 3.11+ because I am using match statements and a lot of typing. If only eda_utils is used, then I think 3.9 is good.

pip install polars scipy nltk scikit-learn xgboost 

(nltk may require additional downloads.)

## TODO

1. Obviously add more useful EDA methods.
2. Build a data preparation pipeline "builder".
3. Common imputation + scaling and their respective.
4. Package this.