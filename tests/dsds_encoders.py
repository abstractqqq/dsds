import pytest
import polars as pl
import dsds.encoders as enc
from polars.testing import assert_frame_equal

@pytest.mark.parametrize("df, cols, drop_first, res", [
    (
        pl.DataFrame({
            "a": ["A", "B", "C", "D", "A", None]
        })
        , ["a"]
        , False
        , pl.DataFrame({
            "a_A":pl.Series("a_A", [1,0,0,0,1,0], dtype=pl.UInt8),
            "a_B":pl.Series("a_B", [0,1,0,0,0,0], dtype=pl.UInt8),
            "a_C":pl.Series("a_C", [0,0,1,0,0,0], dtype=pl.UInt8),
            "a_D":pl.Series("a_D", [0,0,0,1,0,0], dtype=pl.UInt8),
        })
    ),
    (
        pl.DataFrame({
            "a": ["A", "B", "C", "D", "A", None]
        })
        , ["a"]
        , True
        , pl.DataFrame({
            "a_B":pl.Series("a_B", [0,1,0,0,0,0], dtype=pl.UInt8),
            "a_C":pl.Series("a_C", [0,0,1,0,0,0], dtype=pl.UInt8),
            "a_D":pl.Series("a_D", [0,0,0,1,0,0], dtype=pl.UInt8),
        })
    ),
])
def test_one_hot(df:pl.DataFrame, cols:list[str], drop_first:bool, res:pl.DataFrame):
    assert_frame_equal(
        enc.one_hot_encode(df, cols, drop_first=drop_first)
        , res
    )
    assert_frame_equal(
        enc.one_hot_encode(df.lazy(), cols, drop_first=drop_first).collect()
        , res
    )


@pytest.mark.parametrize("df", [
    (pl.DataFrame({
        "a":["a", "b", "c", "a"],
        "b":["y", "n", "y", "n"]
    })
)
])
def test_reverse_one_hot(df:pl.DataFrame):
    assert_frame_equal(
        df, 
        enc.reverse_one_hot_encode(enc.one_hot_encode(df, cols=["a", "b"]), root_col_name=["a", "b"]) 
    )
    assert_frame_equal(
        df, 
        enc.reverse_one_hot_encode(enc.one_hot_encode(df.lazy(), cols=["a", "b"]), root_col_name=["a", "b"]).collect() 
    )


@pytest.mark.parametrize("df, selected, res", [
    (
        pl.DataFrame({
            "a": ["A", "B", "C", "D", "A"],
            "b": ["AA", "BB", "BB", "CC", "CC"]
        })
        ,
        {"a":["A", "B"], "b":["BB"]}
        , 
        pl.DataFrame({
            "a_A":pl.Series("a_A", [1,0,0,0,1], dtype=pl.UInt8),
            "a_B":pl.Series("a_B", [0,1,0,0,0], dtype=pl.UInt8),
            "b_BB":pl.Series("b_BB", [0,1,1,0,0], dtype=pl.UInt8),
        })
    )
])
def test_selective_one_hot(df:pl.DataFrame, selected:dict[str, list[str]], res:pl.DataFrame):
    assert_frame_equal(
        enc.selective_one_hot_encode(df, selected)
        , res
    )
    assert_frame_equal(
        enc.selective_one_hot_encode(df.lazy(), selected).collect()
        , res
    )


@pytest.mark.parametrize("df, cols, n_bins, res", [
    (
        pl.DataFrame({
            "a": range(5)
        })
        , ["a"]
        , 4
        , pl.DataFrame({
            "a":pl.Series("a_A", ["(-inf, 1]","(-inf, 1]","(1, 2]","(2, 3]","(3, inf]"], dtype=pl.Utf8),
        })
    )
])
def test_quantile_binning(df:pl.DataFrame, cols:list[str], n_bins:int, res:pl.DataFrame):
    assert_frame_equal(
        enc.quantile_binning(df, cols, n_bins)
        , res
    )
    assert_frame_equal(
        enc.quantile_binning(df.lazy(), cols, n_bins).collect()
        , res
    )

