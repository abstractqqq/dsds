pub mod metrics;
mod utils;
mod estimates;
mod math;

use polars_core::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PySeries};
use crate::PyReadonlyArray1;

use self::metrics::{
    list_jaccard_similarity,
    series_jaccard_similarity,
    mae,
    mse,
    mape,
    smape,
    huber_loss,
    HashableType
};
use self::estimates::gcc_monte_carlo_prob_est;


#[pyfunction]
pub fn rs_df_inner_list_jaccard(
    pydf: PyDataFrame
    , col_a: &str
    , col_b: &str
    , inner_type: &str
    , include_null:bool
) -> PyResult<PyDataFrame> {

    let df: DataFrame = pydf.into();
    let st: HashableType = HashableType::from_str(inner_type).unwrap();
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

    let st: HashableType = HashableType::from_str(list_type).unwrap();
    let s1: Series = s1.into();
    let s2: Series = s2.into();
    let out: f64 = series_jaccard_similarity(s1, s2, st, include_null, parallel).map_err(PyPolarsErr::from)?;
    Ok(out)

}

#[pyfunction]
pub fn rs_mae(
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

#[pyfunction]
pub fn rs_mse(
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

#[pyfunction]
pub fn rs_mape(
    y_actual:PyReadonlyArray1<f64>,
    y_predicted: PyReadonlyArray1<f64>,
    weighted:bool
) -> f64 {
    let y_a = y_actual.as_array();
    let y_p = y_predicted.as_array();
    mape(y_a, y_p, weighted)
}

#[pyfunction]
pub fn rs_smape(
    y_actual:PyReadonlyArray1<f64>,
    y_predicted: PyReadonlyArray1<f64>,
    double_sum:bool
) -> f64 {
    let y_a = y_actual.as_array();
    let y_p = y_predicted.as_array();
    smape(y_a, y_p, double_sum)
}

#[pyfunction]
pub fn rs_huber_loss(
    y_actual:PyReadonlyArray1<f64>,
    y_predicted: PyReadonlyArray1<f64>,
    delta: f64,
    weights:Option<PyReadonlyArray1<f64>>
) -> f64 {
    let y_a = y_actual.as_array();
    let y_p = y_predicted.as_array();
    match weights {
        Some(we) => huber_loss(y_a, y_p, delta, Some(we.as_array())),
        _ => huber_loss(y_a, y_p, delta, None)
    }
}

#[pyfunction]
pub fn rs_gcc_proba_est(
    n_trials:usize
    , total:u32
    , subsample_amt:u32
    , n_times:u32
    , start_at:u32
    , threshold:f32
) -> u32 {

    let mut n:u32 = start_at;
    let mut p:f32 = 0.;
    while p < threshold {
        n += 1;
        p = gcc_monte_carlo_prob_est(n_trials, total, subsample_amt, n, n_times);
    }
    n
}