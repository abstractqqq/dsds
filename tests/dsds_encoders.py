
import polars as pl
import dsds.encoders as enc
from polars.testing import assert_frame_equal

def test_reverse_one_hot():
    df = pl.DataFrame({
        "a":["a", "b", "c", "a"],
        "b":["y", "n", "y", "n"]
    })

    df2 = enc.reverse_one_hot_encode(enc.one_hot_encode(df, cols=["a", "b"]), root_col_name=["a", "b"])    
    assert_frame_equal(df, df2)

