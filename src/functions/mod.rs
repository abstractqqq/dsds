
pub mod metrics;
pub mod utils;

use polars_core::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PySeries};

use crate::functions::metrics::{
    list_jaccard_similarity,
    series_jaccard_similarity,
    InnerType
};

#[pyfunction]
pub fn rs_df_inner_list_jaccard(
    pydf: PyDataFrame
    , col_a: &str
    , col_b: &str
    , inner_type: &str
    , include_null:bool
) -> PyResult<PyDataFrame> {

    let df: DataFrame = pydf.into();
    let st: InnerType = InnerType::from_str(inner_type).unwrap();
    let out: DataFrame = list_jaccard_similarity(df, col_a, col_b, st, include_null).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(out))

}

#[pyfunction]
pub fn rs_series_jaccard(
    s1: PySeries
    , s2: PySeries
    , list_type: &str
    , include_null: bool
    , parallel: bool    
) -> PyResult<f64> {

    let st: InnerType = InnerType::from_str(list_type).unwrap();
    let s1: Series = s1.into();
    let s2: Series = s2.into();
    let out: f64 = series_jaccard_similarity(s1, s2, st, include_null, parallel).map_err(PyPolarsErr::from)?;
    Ok(out)

}