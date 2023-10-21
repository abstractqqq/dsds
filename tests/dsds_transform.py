import pytest
import polars as pl
import dsds.transform as t
from polars.testing import assert_frame_equal


@pytest.mark.parametrize("df, cols, group_by, strategy, res", [
    (pl.DataFrame({
        "a": [None, 1,2,3,4, None,],
        "b": ["cat", "cat", "cat", "dog", "dog", "dog"],
        "c": [0,0,0,0,1,1]
    })
    , ["a"]
    , ["b", "c"]
    , "mean"
    , pl.DataFrame({
        "a": [1.5, 1,2,3,4, 4],
        "b": ["cat", "cat", "cat", "dog", "dog", "dog"],
        "c": [0,0,0,0,1,1]
    })
    ), 
    (pl.DataFrame({
        "a": [None, 1,2,3,4, None,],
        "b": ["cat", "cat", "cat", "dog", "dog", "dog"],
        "c": [0,0,0,0,1,1]
    })
    , ["a"]
    , ["b", "c"]
    , "max"
    , pl.DataFrame({
        "a": [2,1,2,3,4,4],
        "b": ["cat", "cat", "cat", "dog", "dog", "dog"],
        "c": [0,0,0,0,1,1]
    })
    )
])
def test_hot_deck_impute(df, cols, group_by, strategy, res):
    
    res1 = t.hot_deck_impute(df, cols, group_by, strategy=strategy)
    assert_frame_equal(res1, res)
    res2 = t.hot_deck_impute(df.lazy(), cols, group_by, strategy=strategy).collect()
    assert_frame_equal(res2, res)