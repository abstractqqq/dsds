from typing import Final, Tuple
from .type_alias import PolarsFrame
import polars as pl
from nltk.stem.snowball import SnowballStemmer
from dsds._rust import rs_cnt_vectorizer, rs_get_stem_table, rs_snowball_stem

# Right now, only English. 
# Only snowball stemmer is availabe because I can only find snonball stemmer's implementation in Rust.
# It will take too much effort on my part to add other languages. So the focus is only English for now.

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

def str_col_cleaner(
    cols: list[str],
    replace_by: list[Tuple[str, str]],
    slice: Tuple[int, int],
    lower:bool = True,
    strip:bool = True
):
    pass

def py_count_vectorizer(
    df: pl.DataFrame
    , c: str
    , min_dfreq: float = 0.05
    , max_dfreq: float = 0.95
    , max_word_per_doc: int = 3000
    , max_features: int = 3000
) -> pl.DataFrame:
    
    snow = SnowballStemmer(language="english")
    summary = (
        df.lazy().with_row_count().select(
            pl.col("row_nr")
            , pl.col(c).str.to_lowercase().str.split(" ").list.head(max_word_per_doc)
        ).explode(c)
        .filter((~pl.col(c).is_in(STOPWORDS)) & (pl.col(c).str.lengths() > 2) & (pl.col(c).is_not_null()))
        .select(
            pl.col(c)
            , pl.col(c).apply(snow.stem, return_dtype=pl.Utf8).alias("stemmed")
            , pl.col("row_nr")
        ).groupby("stemmed").agg(
            pl.col(c).unique()
            , doc_freq = pl.col("row_nr").n_unique() / pl.lit(len(df))
        ).filter(
            (pl.col("doc_freq")).is_between(min_dfreq, max_dfreq, closed='both')
        ).top_k(k=max_features, by=pl.col("doc_freq"))
        .select(
            pl.col(c)
            , pl.col("stemmed")
            , pl.col("doc_freq")
        ).sort(by="stemmed").collect()
    )

    exprs = []
    for k,v in zip(summary["stemmed"], summary[c]):
        regex = "(" + "|".join(v) + ")"
        exprs.append(pl.col(c).str.count_match(regex).suffix(f"::cnt_{k}"))

    return df.with_columns(exprs).drop(c)

def count_vectorizer(
    df: PolarsFrame
    , c: str
    , min_dfreq: float = 0.05
    , max_dfreq: float = 0.95
    , max_word_per_doc: int = 3000
    , max_features: int = 500
) -> PolarsFrame:
    
    df_local = df.lazy().select(c).collect()
    ref:pl.DataFrame = rs_get_stem_table(df_local, c, min_dfreq, max_dfreq, max_word_per_doc, max_features)\
                        .sort("stemmed")

    exprs = []
    for s, v in zip(ref.get_column("stemmed"), ref.get_column(c)):
        pattern = f"({'|'.join(v)})"
        exprs.append(
            pl.col(c).str.count_match(pattern).suffix(f"::cnt_{s}")
        )

    return df.with_columns(exprs).drop(c)
