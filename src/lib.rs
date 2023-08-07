use pyo3::prelude::*;
mod linalg;
mod snowball;
mod text;
use numpy::{
    PyReadonlyArray2, 
    PyArray2, 
    ToPyArray
};
use crate::text::text::{
    rs_cnt_vectorizer,
    rs_tfidf_vectorizer,
    rs_ref_table,
    rs_snowball_stem,
    rs_levenshtein_dist,
};
use crate::linalg::metrics::{
    cosine_similarity,
    self_cosine_similarity
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


    #[pyfn(m)]
    fn rs_cosine_similarity<'py>(
        py:Python<'py>,    
        mat1:PyReadonlyArray2<f64>,
        mat2:PyReadonlyArray2<f64>,
        normalize: bool
    ) -> &'py PyArray2<f64> {
        cosine_similarity(
            mat1.as_array().to_owned(),
            mat2.as_array().to_owned(), 
            normalize
        ).to_pyarray(py)
    }
    #[pyfn(m)]
    fn rs_self_cosine_similarity<'py>(
        py:Python<'py>,    
        mat1:PyReadonlyArray2<f64>,
        normalize: bool
    ) -> &'py PyArray2<f64> {
        self_cosine_similarity(mat1.as_array().to_owned(), normalize).to_pyarray(py)
    }

    Ok(())
}