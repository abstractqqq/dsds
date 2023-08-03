use pyo3::prelude::*;
use polars::prelude::*;
pub mod snowball;

pub mod text;
use crate::text::{
    count_vectorizer
};

fn main() {

    let df = CsvReader::from_path("./data/advertising.csv").unwrap().finish().unwrap();
    let out = count_vectorizer(df, "Ad Topic Line", r"[^\s\w\d%]", 0.02, 0.95, 2000).unwrap();

    

    // println!("{:?}", df.head(Some(5)));
}