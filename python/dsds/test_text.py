from typing import Final
from .type_alias import PolarsFrame
import polars as pl
from nltk.stem.snowball import SnowballStemmer

# Right now, settle with NLTK.
# I will look into https://crates.io/crates/rust-stemmers
# I am not confident enough in my Rust to create a rust-Python mix project as of now.
# Using NLTK for now. I see huge potential for speed up as NLTK is mostly
# written in Python.
# I don't expect using Rust to change the code by much.

STOPWORDS:Final[pl.Series] = pl.Series(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves'
             , "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself'
             , 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her'
             , 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them'
             , 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom'
             , 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are'
             , 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having'
             , 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but'
             , 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by'
             , 'for', 'with', 'about', 'against', 'between', 'into', 'through'
             , 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up'
             , 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further'
             , 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all'
             , 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such'
             , 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
             , 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've"
             , 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't"
             , 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn'
             , "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma'
             , 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan'
             , "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't"
             , 'won', "won't", 'wouldn', "wouldn't", 'you'])

def count_vectorizer(
    df: pl.DataFrame
    , c: str
    , tokenizer: str = " "
    , replace: str = '[^\s\w\d%]'
    , min_dfreq: float = 0.05
    , max_dfreq: float = 0.95
    , max_tokens: int = 3000
) -> pl.DataFrame:
    
    snow = SnowballStemmer(language="english")
    summary = (
        df.lazy().with_row_count(name="row_num", offset=1).select(
            pl.col("row_num"),
            pl.col(c).str.replace_all(replace, '').str.to_lowercase().str.split(by=tokenizer).list.head(max_tokens)
        ).explode(c)
        .filter((~pl.col(c).is_in(STOPWORDS)) & (pl.col(c).str.lengths() > 2) & (pl.col(c).is_not_null()))
        .groupby("row_num", c).count().select(
            pl.col("row_num")
            , pl.col(c)
            , pl.col(c).apply(snow.stem, return_dtype=pl.Utf8).alias("stemmed")
            , pl.col("count")/len(df)
        ).groupby("stemmed").agg(
            pl.col(c).unique()
            , pl.col("count").sum()
        ).filter(
            (pl.col("count")).is_between(min_dfreq, max_dfreq, closed='none')
        ).select(
            pl.col(c)
            , pl.col("stemmed")
        ).sort(by="stemmed").collect()
    )

    exprs = []
    for k,v in zip(summary["stemmed"], summary[c]):
        regex = "(" + "|".join(v) + ")"
        exprs.append(pl.col(c).str.count_match(regex).suffix(f"::cnt_{k}"))

    return df.with_columns(exprs).drop(c)