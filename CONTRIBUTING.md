# General Considerations and Guidelines Before Making Contributions:

0. All guidelines below can be discussed and are merely guidelines which may be challenged.

1. If you can read the data into memory, your code should process it faster than Scikit-learn and Pandas. "Abuse" Polars' multi-core capabilities as much as possible before sending data to NumPy.

2. Provide proof that the algorithm generates exact/very close results to Scikit-learn's implementation.

3. Try not to include other core packages. NumPy, Scipy and Polars should be all. The preferred serialization strategy to make things compatible with Polars' execution plan on LazyFrames.

4. Fucntion annotaions are required and functions should have one output type only.

5. Obscure algorithms that do not have a lot of usages should not be included in the package. The package is designed in such a way that it can be customized (A lot more work to be done here.)

Contact me on Discord: t.q