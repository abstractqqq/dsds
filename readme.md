# Categorical Explorative Data Analysis

Some methods I am using to perform categorical explorative data analysis. The goal is to understand "feature importance" without training a model, rank features based on their "importance", understand if variable A is depedent on B or not.

Ideally, this should be applied to datasets with only categorical columns, although I may develop some binning techniques in the future so this should work with continuous variables too. The target column can be a binary classification / multilabel classification target.

My goal is to develop all these test and make them work on a Polars dataframe. A consistent API should be used.

## Available Content

1. A method based on entropy.
    
    Let X denote the test column/feature, and Y the target. We compute the conditional entropy H(Y|X), which represents the remaining randomness in the random variable Y, given the random variable X. For more details, see [here](https://en.wikipedia.org/wiki/Entropy_(information_theory)).

2. Chi-square contigency table test. 
    
    See [here](https://en.wikipedia.org/wiki/Chi-squared_test). G-test is actually used.

## Dependencies

pip install polars, scipy