import polars as pl

def stratified_downsample(df:pl.DataFrame, groupby:list[str], keep:int, keep_pct:float=-1.) -> pl.DataFrame:
    pass