import polars as pl
import graphviz
import polars.selectors as cs

from typing import (
    Optional, 
    Tuple
)
from itertools import combinations
from .prescreen import (
    infer_constants
    , infer_discretes
)
from .type_alias import PolarsFrame

import logging

logger = logging.getLogger(__name__)

# Dataframe comparisons
# Goal:
# 1. Check for similar columns without brute force
# 2. Rank similarity by some stats
# 3. Give user options to remove these 'duplicate columns'
# Leave it here for now.

def _cond_entropy(df: PolarsFrame, x:str, y:str) -> pl.LazyFrame:
    '''
    Computes the conditional entropy H of x given y, usually denoted H(x|y).
    '''

    out = df.lazy().group_by(x, y).agg(
        pl.count()
    ).with_columns(
        (pl.col("count").sum().over(y) / pl.col("count").sum()).alias("prob(y)"),
        (pl.col("count") / pl.col("count").sum()).alias("prob(x,y)")
    ).select(
        pl.lit(x, dtype=pl.Utf8).alias("x"),
        pl.lit(y, dtype=pl.Utf8).alias("y"),
        (-((pl.col("prob(x,y)")/pl.col("prob(y)")).log() 
        * pl.col("prob(x,y)")).sum()).alias("H(x|y)")
    )
    return out
    
def _plot_from_dependency_table(
    df:pl.DataFrame
    , threshold: float 
) -> graphviz.Digraph:
    
    out = df.filter(pl.col("H(x|y)") < threshold)
    remaining_length = len(out)
    i = 0
    frames:list[pl.DataFrame] = [out]
    while remaining_length > 0:
        out = out.join(out, left_on="y", right_on="x", suffix=str(i))
        frames.append(out.clone())
        remaining_length = len(out)
        i += 1

    _ = frames.pop()
    dot = graphviz.Digraph('Dependency Plot', comment=f'Conditional Entropy < {threshold}', format="png") 
    linked = set()
    while len(frames) > 0:
        frame = frames.pop().select(pl.col("x"), pl.col("y"))
        for child, parent in frame.iter_rows():
            if child not in linked:
                linked.add(child)
                dot.node(parent)
                dot.edge(parent, child)
    
    return dot

def dependency_detection(
    df: PolarsFrame
    , cols: Optional[list[str]] = None
    , threshold:float = 0.05
    , plot_tree:bool = True
) -> Tuple[pl.DataFrame, Optional[graphviz.Digraph]]:
    '''
    This method will use `conditional entropy` as a means to meansure dependency between columns. For two discrete
    random variables x, y, the lower the conditional entropy of x given y, denoted H(x|y), the more likely that 
    y determines x. E.g. y = Zipcode, x = State. Then H(x|y) should be low, because knowing the zipcode almost 
    always mean knowing the state.

    This method will return a full table of conditional entropies between all possible pairs of (child, parent)
    columns regardless of threshold, and optionally with a Digraph which is constructed according to the 
    *threshold* given.

    The reason conditional entropies are returned regardless of threshold is that it might be useful in other 
    situations and be an interesting metric to look at. E.g. Conditional entropy is intimately connected to 
    feature importance in decision tree.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe. It is highly recommended that the dataframe is loaded into
        memory.
    cols
        If provided, will use the conditional entropy method to find all dependencies among the given columns.
        If not provided, will use all string columns.
    threshold
        The dependency threshold to be used when constructing the tree.
    plot_tree
        If true, will return a Digraph with the edges connected when conditional entropy is < threshold. You 
        should only turn this off when you want conditional entropy information and runtime is important to you.
    '''
    
    if isinstance(cols, list):
        use_cols = [c for c in cols if c in df.columns]
    else:
        use_cols = df.select(cs.string()).columns # infer_discretes(df)

    df_local = df.select(use_cols)
    constants = infer_constants(df_local)
    if len(constants) > 0:
        logger.info(f"The following columns are not considered because they are constants: {constants}")
        df_local = df_local.select(pl.all().exclude(constants))

    use_cols = df_local.columns
    if len(use_cols) == 0:
        logger.info("All given columns are constants. Nothing is done.")
        return pl.DataFrame()

    n_unique = df_local.lazy().select(
        pl.all().n_unique()
    ).collect().row(0)
    n_unique_asc = sorted(zip(n_unique, use_cols))
    use_cols = [x[1] for x in n_unique_asc]

    frames = (
        _cond_entropy(df_local, x, y) for x, y in combinations(use_cols, 2)
    )
    # Conditional entropy of x given y
    # Because of the arrangement, |x| < |y|, it is impossible that
    # y can be a parent of x.
    conditional_entropy = (
        pl.concat(pl.collect_all(frames))
    )

    if plot_tree:
        tree = _plot_from_dependency_table(conditional_entropy, threshold=threshold)
        return conditional_entropy, tree 
    else:
        return conditional_entropy, None

