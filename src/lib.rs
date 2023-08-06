use pyo3::prelude::*;

pub mod snowball;


pub mod text;
use crate::text::{
    rs_cnt_vectorizer,
    rs_tfidf_vectorizer,
    rs_ref_table,
    rs_snowball_stem,
    rs_levenshtein_dist,
};

// A Python module implemented in Rust.
#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_class::<Test>().unwrap();
    m.add_function(wrap_pyfunction!(rs_cnt_vectorizer, m)?)?;
    m.add_function(wrap_pyfunction!(rs_tfidf_vectorizer, m)?)?;
    m.add_function(wrap_pyfunction!(rs_ref_table, m)?)?;
    m.add_function(wrap_pyfunction!(rs_snowball_stem, m)?)?;
    m.add_function(wrap_pyfunction!(rs_levenshtein_dist, m)?)?;
    Ok(())
}