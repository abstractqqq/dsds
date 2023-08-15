
use polars::prelude::CsvReader;
//use pyo3::prelude::*;
use polars::prelude::*;

use std::time::Instant;

mod snowball;
mod text;
use crate::text::text::get_ref_table;

fn main() {

    println!("hello world!");

    // use rustfft::{FftPlanner, num_complex::Complex};

    // let mut planner = FftPlanner::<f32>::new();
    // let fft = planner.plan_fft_forward(1234);

    // let mut buffer = vec![Complex{ re: 0.0, im: 0.0 }; 1234];

    // fft.process(&mut buffer);

    // println!("{:?}", buffer);
}