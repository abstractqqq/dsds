pub mod text;
mod consts;

use crate::text::text::{
    //count_vectorizer,
    //tfidf_vectorizer,
    //get_ref_table,
    // STEMMER
    // hamming_dist_series,
    snowball_stem,
    hamming_dist,
    levenshtein_dist,
};

// use polars_lazy::prelude::*;
use pyo3::prelude::*;


#[pyfunction]
pub fn rs_hamming_dist(s1:&str, s2:&str) -> Option<u32> {
    hamming_dist(s1, s2)
}

#[pyfunction]
pub fn rs_levenshtein_dist(s1:&str, s2:&str) -> u32 {
    levenshtein_dist(s1, s2)
}

#[pyfunction]
pub fn rs_snowball_stem(word:&str, no_stopwords:bool) -> PyResult<String> {
    let out: Option<String> = snowball_stem(Some(word), no_stopwords);
    if let Some(s) = out {
        Ok(s)
    } else {
        Ok("".to_string())
    }
}