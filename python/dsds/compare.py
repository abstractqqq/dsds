import polars as pl
import polars.selectors as cs
from itertools import product

# Dataframe comparisons
# Goal:
# 1. Check for similar columns without brute force
# 2. Rank similarity by some stats
# 3. Give user options to remove these 'duplicate columns'
# Leave it here for now.


def str_comp(df1:pl.DataFrame, df2:pl.DataFrame):

    df_str1 = df1.select(cs.string())
    df_str2 = df2.select(cs.string())
    for c1, c2 in product(df_str1.columns, df_str2.columns):
        pass 