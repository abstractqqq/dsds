use crate::snowball::{SnowballEnv, algorithms};
use polars_lazy::dsl::GetOutput;
use rayon::prelude::*;
use polars_core::prelude::*;
use polars_lazy::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::PyDataFrame;
use std::iter::zip;

const STOPWORDS:[&str; 179] = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves"
                                , "you're", "you've", "you'll", "you'd", "your", "yours", "yourself"
                                , "yourselves", "he", "him", "his", "himself", "she", "she's", "her"
                                , "hers", "herself", "it", "it's", "its", "itself", "they", "them"
                                , "their", "theirs", "themselves", "what", "which", "who", "whom"
                                , "this", "that", "that'll", "these", "those", "am", "is", "are"
                                , "was", "were", "be", "been", "being", "have", "has", "had", "having"
                                , "do", "does", "did", "doing", "a", "an", "the", "and", "but"
                                , "if", "or", "because", "as", "until", "while", "of", "at", "by"
                                , "for", "with", "about", "against", "between", "into", "through"
                                , "during", "before", "after", "above", "below", "to", "from", "up"
                                , "down", "in", "out", "on", "off", "over", "under", "again", "further"
                                , "then", "once", "here", "there", "when", "where", "why", "how", "all"
                                , "any", "both", "each", "few", "more", "most", "other", "some", "such"
                                , "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"
                                , "s", "t", "can", "will", "just", "don", "don't", "should", "should've"
                                , "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't"
                                , "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn"
                                , "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma"
                                , "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan"
                                , "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't"
                                , "won", "won't", "wouldn", "wouldn't", "you"];


// fn split_offsets(len: usize, n: usize) -> Vec<(usize, usize)> {
//     if n == 1 {
//         vec![(0, len)]
//     } else {
//         let chunk_size = len / n;

//         (0..n)
//             .map(|partition| {
//                 let offset = partition * chunk_size;
//                 let len = if partition == (n - 1) {
//                     len - offset
//                 } else {
//                     chunk_size
//                 };
//                 (partition * chunk_size, len)
//             })
//             .collect()
//     }
// }

#[pyfunction]
pub fn rs_cnt_vectorizer(
    pydf: PyDataFrame
    , c: &str
    , stemmer: &str
    , min_dfreq:f32
    , max_dfreq:f32
    , max_word_per_doc:u32
    , max_feautures: u32
) -> PyResult<PyDataFrame> {

    let df: DataFrame = pydf.into();
    let df: DataFrame = count_vectorizer(df, c, stemmer, min_dfreq, max_dfreq, max_word_per_doc, max_feautures)
                        .map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(df))

}

#[pyfunction]
pub fn rs_levenshtein_dist(s1:&str, s2:&str) -> usize {

    // Naive implementation
    // if s1.is_empty() {
    //     s2.len()
    // } else if s2.is_empty() {
    //     s1.len()
    // } else if s1 == s2 {
    //     0
    // } else {
    //     let s1_owned: String = s1.to_owned();
    //     let s2_owned: String = s2.to_owned();
    //     let (s1_head, s1_tail) = s1_owned.split_at(1);
    //     let (s2_head, s2_tail) = s2_owned.split_at(1);
    //     if s1_head == s2_head {
    //         rs_levenshtein_dist(s1_tail, s2_tail)
    //     } else {
    //         1 + [rs_levenshtein_dist(s1_tail, &s2_owned)
    //             , rs_levenshtein_dist(&s1_owned, s2_tail)
    //             , rs_levenshtein_dist(s1_tail, s2_tail)].iter().min().unwrap()
    //     }
    // }
    // But this is about 10x faster
    // https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm

    let len1: usize = s1.len();
    let len2: usize = s2.len();
    let mut dp: Vec<Vec<usize>> = vec![vec![0; len2 + 1]; len1 + 1];

    // Initialize the first row and first column
    for i in 0..=len1 {
        dp[i][0] = i;
    }

    for j in 0..=len2 {
        dp[0][j] = j;
    }

    // Fill the dp matrix using dynamic programming
    for (i, char1) in s1.chars().enumerate() {
        for (j, char2) in s2.chars().enumerate() {
            if char1 == char2 {
                dp[i + 1][j + 1] = dp[i][j];
            } else {
                dp[i + 1][j + 1] = 1 + dp[i][j].min(dp[i][j + 1].min(dp[i + 1][j]));
            }
        }
    }

    dp[len1][len2]
}


#[pyfunction]
pub fn rs_snowball_stem(word:&str, no_stopwords:bool) -> PyResult<String> {
    let out: Option<String> = snowball_stem(word, no_stopwords);
    if let Some(good) = out {
        Ok(good)
    } else {
        Ok("".to_string())
    }
}

#[pyfunction]
pub fn rs_get_word_cnt_table(
    pydf: PyDataFrame
    , c: &str
    , stemmer: &str
    , min_dfreq:f32
    , max_dfreq:f32
    , max_word_per_doc: u32
    , max_feautures: u32
) -> PyResult<PyDataFrame> {

    let df: DataFrame = pydf.into();
    let out: DataFrame = get_word_cnt_table(df, c,stemmer, min_dfreq, max_dfreq, max_word_per_doc, max_feautures)
                        .map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(out))
}

#[inline]
pub fn snowball_stem(word:&str, no_stopwords:bool) -> Option<String> {

    if word.len() <= 2 {
        None
    } else if (no_stopwords) & (STOPWORDS.contains(&word)) {
        None
    } else {
        let mut env: SnowballEnv<'_> = SnowballEnv::create(word);
        algorithms::english_stemmer::stem(&mut env);
        Some(env.get_current().to_string())
    }
}

#[inline]
fn snowball_on_series(
    words: Series
) -> Result<Option<Series>, PolarsError> {

    Ok(Some(
        words.utf8()?.par_iter()
        .map(|word| {
            match word {
                Some(w) => Ok(snowball_stem(w, true)),
                _ => Ok(None)
            } 
        }).collect::<PolarsResult<Utf8Chunked>>()?.into_series()
    ))
}

#[inline]
pub fn get_word_cnt_table(
    df: DataFrame
    , c: &str
    , stemmer: &str
    , min_dfreq:f32
    , max_dfreq:f32
    , max_word_per_doc: u32
    , max_feautures: u32
) -> PolarsResult<DataFrame> {

    let stemmer_expr:Expr = match stemmer.to_lowercase().as_str() {
        "snowball" => col(c).map(snowball_on_series, GetOutput::from_type(DataType::Utf8)).alias(&"stemmed"),
        _ => col(c).alias(&"stemmed")
    };

    let height: f32 = df.height() as f32;
    Ok(
        df.lazy()
        .with_row_count(&"row_nr", None)
        .select([
            col(&"row_nr")
            , col(c).str().to_lowercase().str().split(&" ").list().head(lit(max_word_per_doc))
        ]).explode([col(c)])
        .filter(col(c).is_not_null())
        .select([
            col(c)
            , stemmer_expr
            , col(&"row_nr")
        ]).groupby([
            col(&"stemmed")
        ]).agg([
            col(c).unique()
            , (col(&"row_nr").n_unique().cast(DataType::Float32) / lit(height)).alias(&"doc_freq")
        ]).filter(
            (col(&"doc_freq").gt_eq(min_dfreq)).and(col(&"doc_freq").lt_eq(max_dfreq))
        ).top_k(max_feautures, [col(&"doc_freq")], [true], true, false)
        .select([
            col("stemmed")
            , col(c)
            , col(&"doc_freq")
        ]).collect()?
    )
}

pub fn count_vectorizer(
    df: DataFrame
    , c: &str
    , stemmer: &str
    , min_dfreq:f32
    , max_dfreq:f32
    , max_word_per_doc: u32
    , max_feautures: u32
) -> PolarsResult<DataFrame> {

    let mut stemmed_vocab: DataFrame = get_word_cnt_table(
                                                df.clone(), 
                                                c,
                                                stemmer,
                                                min_dfreq, 
                                                max_dfreq, 
                                                max_word_per_doc, 
                                                max_feautures
                                        )?.sort(["stemmed"], false, false)?;

    let mut exprs: Vec<Expr> = Vec::with_capacity(stemmed_vocab.height());
    
    let temp: Series = stemmed_vocab.drop_in_place("stemmed")?;
    let stems: &ChunkedArray<Utf8Type> = temp.utf8()?;
    let temp: Series = stemmed_vocab.drop_in_place(c)?;
    let vocabs: &ChunkedArray<ListType> = temp.list()?;

    for (stem, vec) in zip(stems.into_iter(), vocabs.into_iter()) {
        if let Some(w) = stem {
            if let Some(v) = vec {
                let suffix: String = format!("::cnt_{}", w);
                let mut pattern: String = "(".to_string();
                for word in v.utf8()?.into_iter() {
                    if let Some(ww) = word {
                        pattern.push_str(ww);
                        pattern.push('|');
                    }
                }
                pattern.pop();
                pattern.push(')');
                exprs.push(
                    col(c).str().count_match(&pattern).suffix(&suffix)
                )
            }
        }
    }
    let out: DataFrame = df.lazy().with_columns(exprs).drop_columns([c]).collect()?;
    Ok(out)

}