# General Guidelines Before Making Contributions:

0. All guidelines below can be discussed and are merely guidelines which may be challenged.

1. If you can read the data into memory, your code should process it faster than Scikit-learn and Pandas. "Abuse" Polars' multi-core capabilities as much as possible before sending data to NumPy.

2. Provide proof that the algorithm generates exact/very close results to Scikit-learn's implementation.

3. Try not to include other core packages. NumPy, Scipy and Polars should be all. The preferred serialization strategy to make things compatible with Polars' execution plan on LazyFrames.

4. Fucntion annotaions are required and functions should have one output type only.

5. Obscure algorithms that do not have a lot of usages should not be included in the package. The package is designed in such a way that it can be customized (A lot more work to be done here.)

Contact me on Discord: t.q

# Author's Initial Draft of the Purpose of this Package

This library aims to be a lightweight altenative to Scikit-learn (Sklearn), especially in the data preparation stage, e.g. feature screening/selection, basic transformations (scale, impute, one-hot encode, target encode, etc.) Its goal is to replace sklearn's pipeline. (Everything except the models are rewritten. The current dataset builder does not have model steps yet.). Its focuses are on:

1. Being more dataframe centric in design. Dataframe in, dataframe out, and try not to convert or copy to NumPy unless necessary, or provide low-memory options.

2. Performance. Most algorithms are rewritten and are 3-5x faster on large datasets, 10x if you have more cores on your computer, than Scikit-learn's implementation.

3. Simplicity and consistency. This library should not be everything. It should stick to the responsibilities outlined above. It shouldn't become a visualization library. It shouldn't overload users with millions of input  output options, most of which won't be used anyway and which really adds little but side effects to the program. It shouldn't be a package with models. (We might add some wrappers to Scipy for other EDA). This package helps you build and manage the pipeline, from feature selection to basic transformations, and provides you with a powerful builder to build your pipe!

4. Provide more visibility into data pipelines without all the pomp of a web UI. Make data pipelines editable outside Python, which means you can finally copy and paste your pipelines and edit them in a text editor if you want!

5. Be more developer friendly by introducing useful types and data structures in the backend.

To this end, I believe the old "stack", Pandas + Sklearn + some NumPy, is inadequate, mostly because

1. Their lack of parallelism
2. Pandas's "object" types making things difficult and its slow performance.
3. Lack of types enforcement, leading to infinitely many quality checks. Lack of types describing outputs.

Dask and PySpark are distributed systems and so are their own universe. But on a single machine, Polars has proven to be more performant and less memory intensive than both of them.

Most algorithms in Sklearn are available in Scipy, and Scipy relies more heavily on C and usually has multicore options. Therefore, when the algorithm is too complex to perfom in Polars, we can rely on Scipy. 

So the proposed new "stack" is Polars + Scipy + some NumPy.