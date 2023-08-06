from typing import Final, Tuple, Union
from .type_alias import (
    PolarsFrame
    , Stemmer
)
import polars as pl
# from nltk.stem.snowball import SnowballStemmer
from dsds._rust import rs_ref_table, rs_snowball_stem, rs_levenshtein_dist

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

def snowball_stem(word:str, no_stopword:bool=True, language="english") -> str:
    '''
    Stems the word using a snowball stemmer. If you want ultimate speed, use 
    `from dsds._rust import rs_snowball_stem`. This function is merely an ergonomic wrapper
    in Python.

    Parameters
    ----------
    word
        The word to be stemmed
    no_stopword
        If true, English stopwords will be stemmed to the empty string
    language
        Right now English is the only option and the argument will not do anything.
    '''
    return rs_snowball_stem(word, no_stopword)

def levenshtein_dist(s1:str, s2:str) -> int:
    '''
    Computes the Levenshtein distance between two strings. If you want ultimate speed, use 
    `from dsds._rust import rs_levenshtein_dist`. This function is merely an ergonomic wrapper
    in Python.

    Parameters
    ----------
    s1
        The first string
    s2
        The second string
    '''
    return rs_levenshtein_dist(s1,s2)

def clean_str_cols(
    df: PolarsFrame
    , cols: Union[str, list[str]]
    , pattern: str
    , value: str = ""
) -> PolarsFrame:
    '''
    Clean the strings in the given columns by replacing the pattern with the value.

    This will be remembered by blueprint by default.

    Parameters
    ----------
    df
        Either a lazy or eager Polars dataframe
    cols
        Either a string representing a name of a column, or a list of column names
    pattern
        The regex pattern to replace
    value
        The value to replace with
    '''
    if isinstance(cols, str):
        str_cols = [cols]
    else:
        str_cols = cols

    if isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns([pl.col(c).str.replace_all(pattern, value) for c in str_cols])
    return df.with_columns(pl.col(c).str.replace_all(pattern, value) for c in str_cols)

# def py_count_vectorizer(
#     df: pl.DataFrame
#     , c: str
#     , min_dfreq: float = 0.05
#     , max_dfreq: float = 0.95
#     , max_word_per_doc: int = 3000
#     , max_features: int = 3000
# ) -> pl.DataFrame:
    
#     snow = SnowballStemmer(language="english")
#     summary = (
#         df.lazy().with_row_count().select(
#             pl.col("row_nr")
#             , pl.col(c).str.to_lowercase().str.split(" ").list.head(max_word_per_doc)
#         ).explode(c)
#         .filter((~pl.col(c).is_in(STOPWORDS)) & (pl.col(c).str.lengths() > 2) & (pl.col(c).is_not_null()))
#         .select(
#             pl.col(c)
#             , pl.col(c).apply(snow.stem, return_dtype=pl.Utf8).alias("stemmed")
#             , pl.col("row_nr")
#         ).groupby("stemmed").agg(
#             pl.col(c).unique()
#             , doc_freq = pl.col("row_nr").n_unique() / pl.lit(len(df))
#         ).filter(
#             (pl.col("doc_freq")).is_between(min_dfreq, max_dfreq, closed='both')
#         ).top_k(k=max_features, by=pl.col("doc_freq"))
#         .select(
#             pl.col(c)
#             , pl.col("stemmed")
#             , pl.col("doc_freq")
#         ).sort(by="stemmed").collect()
#     )

#     exprs = []
#     for k,v in zip(summary["stemmed"], summary[c]):
#         regex = "(" + "|".join(v) + ")"
#         exprs.append(pl.col(c).str.count_match(regex).suffix(f"::cnt_{k}"))

#     return df.with_columns(exprs).drop(c)

def get_word_cnt_table(
    df: PolarsFrame
    , c: str
    , stemmer:Stemmer = "snowball"
    , min_dfreq: float = 0.05
    , max_dfreq: float = 0.95
    , max_word_per_doc: int = 3000
    , max_features: int = 500
) -> pl.DataFrame:
    '''
    A convenience function that returns the table used to compute word counts. Words with length <= 2 will 
    not be counted. The table has 3 columns:

    (1) A column representing all stems found in the documents in df[c]

    (2) A column representing all words that are mapped to these stems

    (3) Document frequency of the stems

    Parameters
    ----------
    See `dsds.text.count_vectorizer`
    '''
    return rs_ref_table(df.lazy().select(c).collect(), c, stemmer, min_dfreq
                        , max_dfreq, max_word_per_doc, max_features).sort("ref")

def count_vectorizer(
    df: PolarsFrame
    , c: str
    , stemmer:Stemmer = "snowball"
    , min_dfreq: float = 0.05
    , max_dfreq: float = 0.95
    , max_word_per_doc: int = 3000
    , max_features: int = 500
    , lowercase: bool = True
    , persist:bool = False
) -> PolarsFrame:
    '''
    A word count vectorizer similar to sklearn's. In addition, 
    
    (1) It performs stemming and counts the occurrences of all words that are stemmed to the same 
    stem together. It filters out numerics.

    (2) It doesn't convert data to sparse matrix and will output a PolarsFrame.

    If counting for a given list of words is desired, see `dsds.transform.extract_word_count`. Note 
    also that Words of length <=2 will not be counted. See Rust source code for comment on performance.

    Parameters
    ----------
    df
        Either an eager or lazy dataframe. Note that if df is lazy, the column c will be collected.
    c
        Name of the document column
    stemmer
        Only "snowball" stemmer for English is available right now. Everything else will be mapped to no 
        stemmer option.
    min_dfreq
        The minimum document frequency that a word must have. Document Frequency = Sum(Word in Doc) / # Documents
    max_dfreq
        The maximum document frequency above which a word will not be selected.
    max_word_per_doc
        The maximum word count for a document. The document will be truncated after this many words.
    max_features
        The maximum number of word count features to generate. This will take the top words with the highest 
        frequencies
    lowercase
        If true, will lowercase column c first.
    persist
        If df is lazy, this step can be optionally persisted as part of the pipeline (saved in blueprint).
    '''
    if lowercase:
        df_local = df.lazy().with_columns(pl.col(c).str.to_lowercase()).collect()
    else:
        df_local = df.lazy().select(pl.col(c)).collect()
    ref: pl.DataFrame = rs_ref_table(df_local, c, stemmer, min_dfreq
                                    , max_dfreq, max_word_per_doc, max_features).sort("ref")

    exprs = []
    for s, p in zip(ref.get_column("ref"), ref.get_column("captures")):
        exprs.append(
            pl.col(c).str.count_match(p).suffix(f"::cnt_{s}")
        )
    if persist and isinstance(df, pl.LazyFrame):
        return df.blueprint.with_columns(exprs).blueprint.drop([c])
    return df.with_columns(exprs).drop(c)

def tfidf_vectorizer(
    df: PolarsFrame
    , c: str
    , stemmer:Stemmer = "snowball"
    , min_dfreq: float = 0.05
    , max_dfreq: float = 0.95
    , max_word_per_doc: int = 3000
    , max_features: int = 500
    , lowercase: bool = True
    , persist:bool = False
) -> PolarsFrame:
    '''
    A TFIDF vectorizer similar to sklearn's. In addition, 
    
    (1) It performs stemming and counts the occurrences of all words that are stemmed to the same 
    stem together. It filters out numerics. It always computes smooth_idf.

    (2) It doesn't convert data to sparse matrix and will output a PolarsFrame.

    (3) It is a single call. It does not rely on prior count_vectorizer.

    If counting for a given list of words is desired, see `dsds.transform.extract_word_count`. Note 
    also that Words of length <=2 will not be counted. See Rust source code for comment on performance.

    Parameters
    ----------
    df
        Either an eager or lazy dataframe. Note that if df is lazy, the column c will be collected.
    c
        Name of the document column
    stemmer
        Only "snowball" stemmer for English is available right now. Everything else will be mapped to no 
        stemmer option.
    min_dfreq
        The minimum document frequency that a word must have. Document Frequency = Sum(Word in Doc) / # Documents
    max_dfreq
        The maximum document frequency above which a word will not be selected.
    max_word_per_doc
        The maximum word count for a document. The document will be truncated after this many words.
    max_features
        The maximum number of word count features to generate. This will take the top words with the highest 
        frequencies
    lowercase
        If true, will lowercase column c first.
    persist
        If df is lazy, this step can be optionally persisted as part of the pipeline (saved in blueprint).
    '''
    is_lazy = isinstance(df, pl.LazyFrame)
    if lowercase:
        if persist and is_lazy:
            # In this case, persist the lowercase step
            df = df.blueprint.with_columns([
                pl.col(c).str.to_lowercase()
            ])
            # Create local. df has to be lazy
            df_local = df.select(pl.col(c)).collect()
        else: # just lowercase
            df_local = df.lazy().select(pl.col(c).str.to_lowercase()).collect()
    else: 
        df_local = df.lazy().select(pl.col(c)).collect()

    ref: pl.DataFrame = rs_ref_table(df_local, c, stemmer, min_dfreq
                                    , max_dfreq, max_word_per_doc, max_features).sort("ref")

    exprs = []
    for s, p, idf in zip(ref.get_column("ref"), ref.get_column("captures"), ref.get_column("smooth_idf")):
        exprs.append(
            (   pl.lit(idf, dtype=pl.Float64)
                * pl.col(c).str.count_match(p).cast(pl.Float64)
                / pl.col("__doc_len__").cast(pl.Float64)
            ).suffix(f"::tfidf_{s}")
        )
    if persist and is_lazy:
        return (
            df.blueprint.with_columns([
                pl.col(c).str.extract_all(pl.lit(r"(?u)\b\w\w+\b")).list.lengths().cast(pl.Float64).alias("__doc_len__")
            ]).blueprint.with_columns(exprs).blueprint.drop([c, "__doc_len__"])
        )
    else:
        return (
            df.with_columns(
                pl.col(c).str.extract_all(pl.lit(r"(?u)\b\w\w+\b")).list.lengths().cast(pl.Float64).alias("__doc_len__")
            ).with_columns(exprs).drop([c, "__doc_len__"])
        )