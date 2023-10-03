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

use polars_core::prelude::*;

// use polars_lazy::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PySeries};
use rayon::prelude::ParallelIterator;


// Only expose Python Layer in mod.rs, except for things that require py object.

// #[pyfunction]
// pub fn rs_cnt_vectorizer(
//     pydf: PyDataFrame
//     , c: &str
//     , stemmer: &str
//     , min_dfreq:f32
//     , max_dfreq:f32
//     , max_word_per_doc:u32
//     , max_feautures: u32
//     , lowercase: bool
// ) -> PyResult<PyDataFrame> {

//     let df: DataFrame = pydf.into();
//     let st: STEMMER = STEMMER::from_str(stemmer);
//     let df: DataFrame = count_vectorizer(df, c, st, min_dfreq, max_dfreq, max_word_per_doc, max_feautures, lowercase)
//                         .map_err(PyPolarsErr::from)?;
//     Ok(PyDataFrame(df))
// }

// #[pyfunction]
// pub fn rs_tfidf_vectorizer(
//     pydf: PyDataFrame
//     , c: &str
//     , stemmer: &str
//     , min_dfreq:f32
//     , max_dfreq:f32
//     , max_word_per_doc:u32
//     , max_feautures: u32
//     , lowercase: bool
// ) -> PyResult<PyDataFrame> {

//     let df: DataFrame = pydf.into();
//     let st: STEMMER = STEMMER::from_str(stemmer);
//     let df: DataFrame = tfidf_vectorizer(df, c, st, min_dfreq, max_dfreq, max_word_per_doc, max_feautures, lowercase)
//                         .map_err(PyPolarsErr::from)?;
//     Ok(PyDataFrame(df))

// }

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

#[pyfunction]
pub fn rs_snowball_stem_series(words:PySeries, no_stopwords:bool) -> PyResult<PySeries>{
    
    let words: Series = words.into();
    let out = words.utf8()
    .map_err(PyPolarsErr::from)?
    .par_iter()
    .map(|word| {
        snowball_stem(word, no_stopwords)
    }).collect::<ChunkedArray<Utf8Type>>();

    Ok(PySeries(out.into_series()))
}

// #[pyfunction]
// pub fn rs_ref_table(
//     pydf: PyDataFrame
//     , c: &str
//     , stemmer: &str
//     , min_dfreq:f32
//     , max_dfreq:f32
//     , max_word_per_doc: u32
//     , max_feautures: u32
// ) -> PyResult<PyDataFrame> {
    
//     // get_ref_table assumes all docs in df[c] are already lowercased

//     let df: DataFrame = pydf.into();
//     let st: STEMMER = STEMMER::from_str(stemmer);
//     let out: DataFrame = get_ref_table(df, c, st, min_dfreq, max_dfreq, max_word_per_doc, max_feautures)
//                         .map_err(PyPolarsErr::from)?;
//     Ok(PyDataFrame(out))
// }