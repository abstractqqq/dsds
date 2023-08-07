
use polars::prelude::CsvReader;
//use pyo3::prelude::*;
use polars::prelude::*;

use std::time::Instant;

mod snowball;
mod text;
use crate::text::text::get_ref_table;

fn main() {

    let n: usize = 20;
    let mut times:Vec<f32> = Vec::with_capacity(n);
    let df: DataFrame = CsvReader::from_path("./data/train.csv").unwrap().finish().unwrap();
    for _ in 0..n {
        let start = Instant::now();


        let _table = get_ref_table(df.clone(), "Description", "snowball", 0.02, 0.95, 1000, 200).unwrap();

        // println!("{}", table.unwrap().head(Some(5)));

        let duration: f32 = start.elapsed().as_secs_f32();
        times.push(duration);
    }

    let avg = times.into_iter().sum::<f32>() / n as f32; 
    println!("Average time: {}", avg);

    
}