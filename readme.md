# My Data Analysis Toolkit

Goal1: Make traditional EDA methods (especially those in sklearn) easier to perform, faster to compute, and more "DataFrame-friendly", meaning that 

1. Inputs should be dataframes and reduce copying to NumPy array as much as possible, instead of inputting NumPy arrays or internally copying data to numpy array.
2. Output should be clean dataframes on which we can quickly sort and filter.

To this end, we have to entirely use Polars as Pandas is not a great performance-wise and it has inconsistent data type issues. 

Goal2: Create memory efficient way of scaling, transforming data (DataFrame -> NumPy). Speed may be compromised in the interest of low memory.

Goal3: Dataframe-friendly text transformations, good for small scale NLP analysis. May not be efficient over large datasets.


## Explorative Data Analysis (eda.py)

My goal is to develop all these test and make them work on a Polars dataframe. A consistent API should be used. These are the methods I have so far.

0. Feature removal methods (null_pct, var, etc.) and binary transforms. More in the future.

1. A method based on entropy. (Univariate as of now, but can be potentially multi-variate.)
    
    Let X denote the test column/feature, and Y the target. We compute the conditional entropy H(Y|X), which represents the remaining randomness in the random variable Y, given the random variable X. For more details, see [here](https://en.wikipedia.org/wiki/Entropy_(information_theory)).

2. Classic Anova One Way F-test. (Univariate) 
    
    See [here](https://saylordotorg.github.io/text_introductory-statistics/s15-04-f-tests-in-one-way-anova.html).

## Text data transformation (text_data.py)

1. A quick reusable method to transform text data in a column into multiple numerical columns. Great for TF-IDF kind of thing and also any other model that cannot consume text data directly. 

## Dependencies

Python 3.9+

pip install polars scipy nltk scikit-learn

(nltk may require additional downloads.)

## TODO

1. Obviously make more useful EDA methods "DataFrame-friendly."