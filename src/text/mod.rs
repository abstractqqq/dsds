pub mod text;
mod consts;

use crate::text::text::{
    count_vectorizer,
    tfidf_vectorizer,
    snowball_stem,
    get_ref_table
};
use polars_core::prelude::*;
// use polars_lazy::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::PyDataFrame;

// Only expose Python Layer in mod.rs

#[pyfunction]
pub fn rs_cnt_vectorizer(
    pydf: PyDataFrame
    , c: &str
    , stemmer: &str
    , min_dfreq:f32
    , max_dfreq:f32
    , max_word_per_doc:u32
    , max_feautures: u32
    , lowercase: bool
) -> PyResult<PyDataFrame> {

    let df: DataFrame = pydf.into();
    let df: DataFrame = count_vectorizer(df, c, stemmer, min_dfreq, max_dfreq, max_word_per_doc, max_feautures, lowercase)
                        .map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(df))
}

#[pyfunction]
pub fn rs_tfidf_vectorizer(
    pydf: PyDataFrame
    , c: &str
    , stemmer: &str
    , min_dfreq:f32
    , max_dfreq:f32
    , max_word_per_doc:u32
    , max_feautures: u32
    , lowercase: bool
) -> PyResult<PyDataFrame> {

    let df: DataFrame = pydf.into();
    let df: DataFrame = tfidf_vectorizer(df, c, stemmer, min_dfreq, max_dfreq, max_word_per_doc, max_feautures, lowercase)
                        .map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(df))

}

#[pyfunction]
pub fn rs_hamming_dist(s1:&str, s2:&str) -> Option<usize> {
    if s1.len() != s2.len() {
        return None
    }
    Some(
        s1.chars().zip(s2.chars()).fold(
            0, |acc, (c1,c2)| acc + (c1 != c2) as usize
        )
    )
}

#[pyfunction]
pub fn rs_levenshtein_dist(s1:&str, s2:&str) -> usize {

    // https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm

    let (len1, len2) = (s1.len(), s2.len());
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
    if let Some(good) = snowball_stem(word, no_stopwords) {
        Ok(good)
    } else {
        Ok("".to_string())
    }
}

#[pyfunction]
pub fn rs_ref_table(
    pydf: PyDataFrame
    , c: &str
    , stemmer: &str
    , min_dfreq:f32
    , max_dfreq:f32
    , max_word_per_doc: u32
    , max_feautures: u32
) -> PyResult<PyDataFrame> {
    
    // get_ref_table assumes all docs in df[c] are already lowercased

    let df: DataFrame = pydf.into();
    let out: DataFrame = get_ref_table(df, c,stemmer, min_dfreq, max_dfreq, max_word_per_doc, max_feautures)
                        .map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(out))
}