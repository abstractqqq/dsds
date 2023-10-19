# The Problem with Feature Selection 


## Model-based Methods

We have all used `.feature_importances_` some time in our career as a measure of feature importance. But there are some fatal flaws:

1. A bad model will not correctly use the features, will likely mistake lead to a bad evaluation of the importance of the features.
2. But if we have a good model, why do we still care about feature selection?
3. Training a good model just for feature selection seems to be a waste of time.
4. Most tree-based models use feature space subsampling. What happens when there are multiple highly correlated features?

In addition, SHAP importance suffers from the same problem as feature importance because of its reliance on a good model. Simply swaping feature importance out for SHAP will not really solve the issues.

## Univariate Methods

1. We also have methods like Anova F-score, Mutual Information Score, KS-statistics, Absolute correlation, Weight of Evidence, etc. But they only consider how each feature interacts with the target, but not how combinations of the features impact the target. 

# A Reasonable Solution

First, let's understand why in practice we need feature selection.

## Understanding Feature Selection from a Practical Point of View

1. Real life datasets can have thousands of columns (features). But a lot of the features are either useless, has terrible quality, or highly correlated with each other.
2. Training a model with thousands of features will lead to
    1. Longer training time.
    2. Very hard to maintain and monitor the drift of these features over time. What happens if a few features are suddenly behaving differently from what they were like in the training dataset?
    3. A lot of features are bought from data vendors in real life. So there is a cost concern.
    4. More features typically mean more processing. E.g cleaning, scaling, transformations, imputing, encoding for string features. Even worse, for string features, the model may like one-hot-encoding and for others the model may prefer target encoding etc. The more processing steps, the harder it is to find the proper handling of the features and the more latency the model would introduce in production.

Genenerally speaking, practitioners want to keep things simple. How can we quickly go from thousands of features to, say, 100 features? As I pointed out above, maintaining and monitoring a high feature model is a pain. 

**I believe, and I believe I am not the only one who thinks so, the essence of feature selection in modern machine learning workflow is not about picking out the best features, but more about picking out good and reasonable features so that the project can move quickly and achieve the targets/goals.** 

To this end, I will present my solution to the problems listed above.

1. If model-based importance is desired, we should use a fast tuner to train a model. We certainly do not have time to do a grid search during feature selection. But tools like Optuna enables us to tune the model well relatively fast. I chose Optuna for DSDS because I think Optuna is light-weight and easier to configure. A 30-trial tuning usually yields an ok model and gives us pretty good hyperparameters. LBGM is a top model and faster than XGBoost and has similar performance in most cases. So the most suitable model is LGBM. By quickly tuning a LGBM, we can get a more reliable sense of the feature importance. With this, we can finish a detailed model-based feature selection workflow in just a few hours, or at most one business day.

2. [MRMR](https://en.wikipedia.org/wiki/Minimum_redundancy_feature_selection). Maximum Relevance Minimum Redundancy technique not only helps us knock out highly correlated features, but it also helps us shrink the size of the features to a desired number. My package, DSDS has the best in class implementation of MRMR, which greatly reduces the processing time needed for MRMR and has the same output as the most popular MRMR package right now. (Screenshot below is a dataset with 50k rows and 500 columns. It is very common for bank/credit model datasets to have 1000+ columns.)![mrmr](./pics/mrmr.PNG)


So given all these, we have multiple strategies to run MRMR. We can use the univariate metrics as relevance, or we can use feature_importance given by LGBM with hyperparameters tuned (Not perfectly tuned) by Optuna. With these various methods, we can create 1-3 feature sets. Personally, I would usually go for ~50 or ~100 and start iterating. 

The benefit of doing so is that now feature size is small enough and it is now reasonable to run SHAP or other model case-specific methods to do feature "refinement", more detailed analysis on impact of individual features on models, and better filtering. Since we created multiple feature sets, we can compare and swap features among them without worrying about missing good features.


## A Word on String Features

You can potentially encode them and treat them as numerical features and run the above procedure to determine their importance. But the question is how do you know the best encoding for such features? Sometimes, model likes it to be one-hot, sometimes Target encoded. There is really no good way. Therefore, I personally prefer separating numerical feature selection from string feature selection.

For string feature selection, we can use discrete information gain, or WOE information value as metrics.