import polars as pl 
import pandas as pd 
from typing import Tuple
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# nltk.download("punkt")
stop = stopwords.words('english')

def _reverse_memo(memo:dict[str, str]) -> dict[str, list[str]]:
    '''
    Reverse the memo used in transform_text_data. 
        
    Returns: 
        the mapping between stemmed words and the many versions of the word that get stemmed to the same "root".
    
    '''
    output:dict[str, list[str]] = {}
    for key, item in memo.items():
        if item in output:
            output[item].append(key)
        else:
            output[item] = [key]

    return output 

def transform_text_data(df:pl.DataFrame|pd.DataFrame
    , text_cols:list[str]
    , vectorize_method:str = "count"
    , max_df:float=0.95, min_df:float=0.05,
) -> Tuple[pl.DataFrame, dict[str, list[str]]]:
    ''' 
    Given a dataframe, perform one-hot encoding and TFIDF/count transformation for the respective columns.
    This may not be optimized.

    Arguments:
        df: input Pandas/Polars dataframe
        text_cols: a list of str representing column names that need to be TFIDF/count vectorized
        vectorizer: str, either "count" or "tfidf"
        max_df: do not consider words with document frequency > max_df. Default 0.95.
        min_df: do not consider words with document frequency < min_df. Default 0.05.

    Returns:
        transformed polars dataframe, and a memo of what is being mapped to what.

    '''

    df2 = df.with_columns((
        # first step in cleaning the data, perform split, so now it is a []
        pl.col(t).str.replace_all('[^\s\w\d%]', '').str.to_lowercase().str.split(by=" ")
        for t in text_cols
    ))

    print("Perfoming stemming...")
    ps = PorterStemmer()
    memo = {} # perform memoization for stemming
    if vectorize_method == "tfidf":
        vectorizer:TfidfVectorizer = TfidfVectorizer(max_df = max_df, min_df = min_df, stop_words=list(stop))
    else:
        vectorizer:CountVectorizer = CountVectorizer(max_df = max_df, min_df = min_df, stop_words=list(stop))
    
    new_columns = [df2] # df2 will be changed in the for loop, but that is ok.
    for c in text_cols:
        new_list:list[str] = []
        text_column = df2.drop_in_place(c)
        for tokens in text_column:
            if not tokens.is_empty(): # if tokens is not empty. Tokens is a pl.Series.
                new_tokens:list[str] = [] 
                for w in tokens:
                    if w not in memo:
                        memo[w] = ps.stem(w)
                    new_tokens.append(memo[w])
                # re-map the tokens into sentences, in order to use scikit-learn builtin vectorize methods. Bad for performance? Can this be improved?
                new_list.append(" ".join(new_tokens))
            else:
                new_list.append("Unknown")

        print(f"Performing {vectorize_method.capitalize()} vectorization for {c}...")
        # Vectorizer will return Sparse matrix
        X:csr_matrix = vectorizer.fit_transform(new_list)
        names:list[str] = [c+"::word::"+x for x in vectorizer.get_feature_names_out()]
        # Add this new dataframe to a list 
        new_columns.append(pl.from_numpy(X.toarray(), schema=names))

    df_final = pl.concat(new_columns, how="horizontal")
    return df_final, _reverse_memo(memo)