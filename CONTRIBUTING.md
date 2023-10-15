# General Guidelines Before Making Contributions:

0. All guidelines below can be discussed and are merely guidelines which may be challenged.

1. If you can read the data into memory, your code should process it faster than Scikit-learn and Pandas. "Abuse" Polars' multi-core capabilities as much as possible. Send data to NumPy as a last resort. (Exception being Matrix stuff, but I am considering using more Rust for that use case.)

2. Provide proof that the algorithm generates exact/very close results to Scikit-learn's implementation. I am in the process of building out a test suite. It is not the most fun part. But generally I accept a "It ran in Jupyter notebook" screeshot. But if more people are collaborating, it is definitely good if we can check the code runs on each other's machines. Again, complete test and robust CI is in the plan and someone has to take the time...

3. Try not to include other core packages. NumPy, Scipy and Polars should be all. One day, we might want to remove SciPy and NumPy if we can rely on Rust more.

4. Fucntion annotaions, types, are required and functions should have one output type only.

5. If the function takes a dataframe in and returns a dataframe out, it should work for lazy and eager dataframes. (For most of the cases.)

Contact me on Discord: t.q

Put `#[Rustify]` on top the function if you think we should Rustify this function.

# So, what am I working on what am I planning?

You are welcomed to work on any of the areas mentioned below.

The major focus is in:

1. Prescreen, aka what kind of data am I dealing with? Data set diagonsis. Which columns are dependent on which? Which columns are highly correlated?
2. Feature selection. Obviously we need more methods. But feature selection has always been a black box. We might want to consolidate some options. Should we use LightGBM's feature_importances for all algorithms? The choices are limited and other choices seem to lead to slower runtime and usually worse results. Model-based feature selection also has its own problems. E.g. how can you trust the features coming from a bad model? But if I have a good model, why am I still doing feature selection?
3. Better pipeline management. See blueprint. This will make custom transformers trivial with Polars expressions.
4. Performance improvement.

There are some new areas that the package is lacking in:

1. Sampling strategies. (Basic ones are done)
2. Splitting strategies.
3. Matrices, PCA, and some other more analytical methods that rely more on linear algebra. Solvers. Linear programming. I am leaning towards Faer-rs for more linear algebra stuff. But right now interoperability is lacking.
4. Time series. There is a dedicated time series analysis package being build called functime. I am also a contributor to that project and wrote lots of time series feature extraction methods. DSDS will not focus too much on time series.
5. Clustering. I don't think it is too hard to integrate more cutting edge clustering algorithms like [k-medoids.](https://github.com/kno10/rust-kmedoids)

There are some abandoned effort due to scope and limited time:

1. Cleaning and processing text data. Vectorization of text data. This was done, but I reverted it. This by itself can be a huge module! Polars has great text support. I looked into Rust Tokenizers and it seems very easy to integrate it into the current package. But since I don't work with language models that often, I am not sure about the design of the API and therefore have been hesitant to implement.

There are some interesting ideas:

1. Add a trading/time series module. Polars make technical indicators easy and fast to compute, especially with common subexpression elimination!!! This is likely more proper for functime. I am in constant communication with functime's owner.
2. Add a dataframe comparison module. This is part of prescreen, but I think the content deserve its own module. This will be hard. The goal of this module will be to identify similar/same columns that are "shuffled" because they come from some join operation. Identify data dependency issues. This is a big issue for large companies with tons of data.
3. A better, deeper, more useful data profiling methods. We can profile data using many statistics, then use k-medoids to cluster them! Or use cosine similarity or something to see how close they are! This will be useful in duplicate detection, or other data mapping problems.
4. Basic Image processing. Very basic ones.
5. Some data comes with graph structures (with join columns connecting them together), and therefore we can implement some basic graph algorithms (likely in Rust).