# General Guidelines Before Making Contributions:

0. All guidelines below can be discussed and are merely guidelines which may be challenged.

1. If you can read the data into memory, your code should process it faster than Scikit-learn and Pandas. "Abuse" Polars' multi-core capabilities as much as possible before sending data to NumPy.

2. Provide proof that the algorithm generates exact/very close results to Scikit-learn's implementation.

3. Try not to include other core packages. NumPy, Scipy and Polars should be all. One day, we might want to remove scipy if we can rely on Rust more.

4. Fucntion annotaions are required and functions should have one output type only.

5. What you write should work for lazy and eager dataframes. If you want it to work with pipeline, there are some specific things you need to do. Contact me for details.

Contact me on Discord: t.q

# Initial Draft of the Purpose of this Package

This library aims to be a lightweight altenative to Scikit-learn (Sklearn), especially in the data preparation stage, e.g. feature screening/selection, basic transformations (scale, impute, one-hot encode, target encode, etc), and some other tasks which traditionally was done via Python UDFs. Its goal is to replace sklearn's pipeline. (Everything except the models are rewritten. The current dataset builder does not have model steps yet.). Its focuses are on:

1. Being more dataframe centric in design. Dataframe in, dataframe out, and try not to convert or copy to NumPy unless necessary, or provide low-memory options.

2. Performance. Most algorithms are rewritten and are 3-5x faster on large datasets, 10x if you have more cores on your computer, than Scikit-learn's implementation.

3. Simplicity and consistency. This library should not be everything. It should stick to the responsibilities outlined above. It shouldn't become a visualization library. It shouldn't overload users with millions of input/output options, most of which won't be used anyway. Side effects should be minimized. This package helps you build and manage the pipeline, from feature selection to basic transformations.

4. Be more developer friendly by introducing useful types and data structures in the backend.

To this end, I believe the old "stack", Pandas + Sklearn + some NumPy, is inadequate, mostly because

1. Their lack of speed and parallelism
2. Pandas's "object" types making things difficult and its slow performance.
3. Lack of types enforcement, leading to infinitely many type checks. Hard to develop with.

PySpark is a distributed system and has its own universe, and I don't think people should do traditional ML in PySpark environment. On a single machine, Polars has proven to be more performant and less memory intensive than alternatives.

# So, what am I working on what am I planning?

You are welcomed to work on any of the areas mentioned below.

The major focus is in:

1. Prescreen, aka what kind of data am I dealing with?
2. Feature selection. Obviously we need more methods. But feature selection has always been a black box. We might want to consolidate some options. Should we use LightGBM's feature_importances for all algorithms? The choices are limited and other choices seem to lead to slower runtime and usually worse results.
3. Improve developer experience with dsds's blueprint (Pipeline) and making pipelines more useful.
4. Performance improvement.

There are some new areas that the package is lacking in:
1. Sampling strategies. (Basic ones are done)
2. Splitting strategies.
3. Writing useful model evaluation metrics from scratch.

There are some abandoned effort due to scope and limited time:
1. EDA, prescreen for text data. This by itself can be a huge module! Polars has great text support.

There are some interesting ideas:
1. Add a trading/time series module. Polars make technical indicators easy and fast to compute!!!
2. Add a dataframe comparison module. This is part of prescreen, but I think the content deserve its own module. This will be hard. The goal of this module will be to identify similar/same columns that are "shuffled" because they come from some join operation. This is a big issue for large companies with tons of data.