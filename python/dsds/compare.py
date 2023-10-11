import polars as pl
import graphviz
import polars.selectors as cs
import logging

from typing import (
    Optional, 
    Tuple
)
from itertools import combinations
from .prescreen import (
    get_unique_count
    # , infer_discretes
)
from .type_alias import PolarsFrame


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

    out = df.lazy().group_by(pl.col(x), pl.col(y)).agg(
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
    
    # Filter
    out = df.filter(pl.col("H(x|y)") < threshold).select(
        pl.col('x').alias("child"), # c for child
        pl.col('y').alias("parent") # p for parent
    )
    cp = out.group_by("child").agg(pl.col("parent"))
    pc = out.group_by("parent").agg(pl.col("child"))
    child_parent:dict[str, pl.Series] = dict(zip(cp.drop_in_place("child"), cp.drop_in_place("parent")))
    parent_child:dict[str, pl.Series] = dict(zip(pc.drop_in_place("parent"), pc.drop_in_place("child")))

    dot = graphviz.Digraph('Dependency Plot', comment=f'Conditional Entropy < {threshold:.2f}', format="png") 
    for c, par in child_parent.items():
        parents_of_c = set(par)
        for p in par:
            # Does parent p have a child that is also a parent of c?
            # If so, remove p.
            children_of_p = parent_child.get(p, None)
            if children_of_p is not None:
                if len(parents_of_c.intersection(children_of_p)) > 0:
                    parents_of_c.remove(p)

        dot.node(c)
        for p in parents_of_c:
            dot.node(p)
            dot.edge(p, c)

    return dot

def dependency_violation(
    df: PolarsFrame
    , conditional_entropy: pl.DataFrame
    , threshold: float = 0.05
) -> list[pl.DataFrame]:
    '''
    Given conditional entropy, run a diagnosis and find problematic data points in the original df. This
    will return a list of dataframes, with each representing

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe. It is highly recommended that the dataframe is loaded into
        memory.
    conditional_entropy
        A dataframe with the same schema as the output of dependency_detection call.
    threshold
        The threshold to use to confirm dependency. The lower the stricter.
    '''
    
    possible_violations = conditional_entropy.filter(
        pl.col("H(x|y)").is_between(0, threshold, closed="none")
    )
    child = possible_violations.drop_in_place("x")
    parent = possible_violations.drop_in_place("y")

    frames = (
        df.lazy().group_by(p).agg(
            pl.count(),
            pl.col(c).value_counts()
        ).filter(
            pl.col(c).list.len() > 1
        ).select(
            pl.col(p).alias(f"Column Name: {p}"),
            pl.col("count"),
            pl.col(c).alias(f"Col {c}: Value & Count"),
            pl.lit(f"`{p}` should uniquely determine `{c}`").alias("Reason")
        )
        for c, p in zip(child, parent)
    )
    out = pl.collect_all(frames)
    return out

def dependency_detection(
    df: PolarsFrame
    , cols: Optional[list[str]] = None
    , threshold:float = 0.05
    , plot_tree:bool = True
) -> Tuple[pl.DataFrame, Optional[graphviz.Digraph]]:
    '''
    This method will use `conditional entropy` to meansure dependency between columns. For two discrete
    random variables x, y, the lower the conditional entropy of x given y, denoted H(x|y), the more likely that 
    y determines x. E.g. y = Zipcode, x = State. Then H(x|y) should be low, because knowing the zipcode almost 
    always mean knowing the state.

    This method will return a full table of conditional entropies between all possible pairs of (child, parent)
    columns regardless of threshold, and optionally with a Digraph which is constructed according to the 
    *threshold* given. Right now only one potential parent of each node is shown in the graph. Other parents can
    be inferred from the first dataframe output. I will update this when I figure out how to do MST efficiently.

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
        The threshold to use to confirm dependency. The lower the stricter. It will also be used when 
        constructing the dependency tree.
    plot_tree
        If true, will return a Digraph with the edges connected when conditional entropy is < threshold. You 
        should only turn this off when you want conditional entropy information and runtime is important to you.
    '''
    
    if isinstance(cols, list):
        use_cols = [c for c in cols if c in df.columns]
    else:
        use_cols = df.select(cs.string()).columns # infer_discretes(df)

    df_local = df.select(use_cols)
    n_unique = get_unique_count(df_local, False).sort("n_unique").set_sorted("n_unique")
    constants = n_unique.filter(pl.col("n_unique") == 1)["column"].to_list()
    if len(constants) > 0:
        logger.info(f"The following columns are not considered because they are constants: {constants}")
        df_local = df_local.select(pl.all().exclude(constants))

    # already sorted
    n_unique_nonconst = n_unique.filter(pl.col("n_unique") > 1)
    use_cols = n_unique_nonconst["column"].to_list()
    df_local = df_local.select(use_cols)
    if len(use_cols) == 0:
        logger.info("No available column. Either there is no string in the dataframe, or there is no column "
                    "provided by the user, or all columns are constant. Nothing is done.")
        return pl.DataFrame(), None
    
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

