use pyo3::prelude::*;
//use pyo3::exceptions::PyValueError;
mod functions;
mod snowball;
mod text;
use numpy::{
    // PyReadonlyArray1,
    PyReadonlyArray2, 
    PyArray2, 
    IntoPyArray, PyReadonlyArray1
};
use crate::text::{
    rs_cnt_vectorizer,
    rs_tfidf_vectorizer,
    rs_ref_table,
    rs_snowball_stem,
    rs_snowball_stem_series,
    rs_levenshtein_dist,
    rs_hamming_dist,
    rs_hamming_dist_series
};
use crate::functions::{
    rs_df_inner_list_jaccard,
    rs_series_jaccard, 
    metrics::{  
        cosine_similarity,
        self_cosine_similarity,
        mae,
        mse,
        mape
    }
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
    m.add_function(wrap_pyfunction!(rs_df_inner_list_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(rs_series_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(rs_hamming_dist, m)?)?;
    m.add_function(wrap_pyfunction!(rs_snowball_stem_series, m)?)?;
    m.add_function(wrap_pyfunction!(rs_hamming_dist_series, m)?)?;

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
        ).into_pyarray(py)
    }

    #[pyfn(m)]
    fn rs_self_cosine_similarity<'py>(
        py:Python<'py>,    
        mat1:PyReadonlyArray2<f64>,
        normalize: bool
    ) -> &'py PyArray2<f64> {
        self_cosine_similarity(mat1.as_array().to_owned(), normalize).into_pyarray(py)
    }

    #[pyfn(m)]
    fn rs_mae(
        y_actual:PyReadonlyArray1<f64>,
        y_predicted: PyReadonlyArray1<f64>,
        weights:Option<PyReadonlyArray1<f64>>
    ) -> f64 {
        let y_a = y_actual.as_array();
        let y_p = y_predicted.as_array();

        match weights {
            Some(we) => {
                let w = we.as_array();
                mae(y_a, y_p, Some(w))
            },
            _ => mae(y_a, y_p, None)
        }
    }

    #[pyfn(m)]
    fn rs_mse(
        y_actual:PyReadonlyArray1<f64>,
        y_predicted: PyReadonlyArray1<f64>,
        weights:Option<PyReadonlyArray1<f64>>
    ) -> f64 {
        let y_a = y_actual.as_array();
        let y_p = y_predicted.as_array();
        match weights {
            Some(we) => {
                let w = we.as_array();
                mse(y_a, y_p, Some(w))
            },
            _ => mse(y_a, y_p, None)
        }
    }

    #[pyfn(m)]
    fn rs_mape(
        y_actual:PyReadonlyArray1<f64>,
        y_predicted: PyReadonlyArray1<f64>,
        weighted:bool
    ) -> f64 {
        let y_a = y_actual.as_array();
        let y_p = y_predicted.as_array();
        mape(y_a, y_p, weighted)
    }

    Ok(())
}