from typing import Final, Union, Optional
from .type_alias import (
    PolarsFrame
    # , Stemmer
)
# from .blueprint import (
#     _dsds_with_columns
# )
from dsds._dsds_rust import (
    # rs_ref_table, 
    rs_snowball_stem,
    rs_levenshtein_dist,
    rs_hamming_dist,
)
import polars as pl

# Right now, only English. 
# Only snowball stemmer is availabe because I can only find snonball stemmer's implementation in Rust.
# It will take too much effort on my part to add other languages. So the focus is only English for now.

STOPWORDS:Final[pl.Series] = pl.Series(["a", "about", "above", "after", "again", "against", "ain", 
            "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", 
            "at", "be", "because", "been", "before", "being", "below", "between", 
            "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", 
            "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", 
            "down", "during", "each", "few", "for", "from", "further", "had", "hadn", 
            "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", 
            "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", 
            "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", 
            "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", 
            "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", 
            "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", 
            "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", 
            "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", 
            "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", 
            "then", "there", "these", "they", "this", "those", "through", "to", "too", 
            "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", 
            "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", 
            "why", "will", "with", "won't", "wouldn", "wouldn't", "y", "you", "you'd", 
            "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"])

def snowball_stem(word:str, no_stopword:bool=True, language="english") -> str:
    '''
    Stems the word using a snowball stemmer. If you want ultimate speed, use 
    `from dsds._rust import rs_snowball_stem`. You will have to supply a str and a bool 
    every time you call the rs_snowball_stem but it is the fasteest. This function is merely 
    an ergonomic wrapper in Python.

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

def hamming_dist(s1:str, s2:str) -> Optional[int]:
    '''
    Computes the hamming distance between two strings. If you want ultimate speed, use 
    `from dsds._rust import rs_hamming_dist`. This function is merely an ergonomic wrapper
    in Python. If s1 and s2 do not have the same length, None will be returned.
    '''
    return rs_hamming_dist(s1,s2)

def levenshtein_dist(s1:str, s2:str) -> int:
    '''
    Computes the Levenshtein distance between two strings. If you want ultimate speed, use 
    `from dsds._rust import rs_levenshtein_dist`. This function is merely an ergonomic wrapper
    in Python.
    '''
    return rs_levenshtein_dist(s1,s2)


