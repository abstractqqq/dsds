use pyo3::prelude::*;

pub mod snowball;

pub mod text;
use crate::text::{
    rs_cnt_vectorizer,
    rs_get_stem_table,
    rs_snowball_stem
};

/// A Python module implemented in Rust.
#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_class::<Test>().unwrap();
    m.add_function(wrap_pyfunction!(rs_cnt_vectorizer, m)?)?;
    m.add_function(wrap_pyfunction!(rs_get_stem_table, m)?)?;
    m.add_function(wrap_pyfunction!(rs_snowball_stem, m)?)?;
    Ok(())
}