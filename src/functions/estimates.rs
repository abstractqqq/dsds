use ndarray::Array1;
use rand::thread_rng;
use rand::seq::SliceRandom;
use rayon::prelude::*; 

// Sample pool for the Generalized Coupon Collector's problem.
fn gcc_sample_pooled(
    total:u32
    , subsample_amt:u32
    , subsample_times:u32
) -> Array1<u32>{
    let total_usize = total as usize;

    let mut count:Array1<u32> = Array1::from_elem(total_usize, 0);
    let mut vec:Vec<u32> = (0..total).collect();
    for _ in 0..subsample_times {
        vec.shuffle(&mut thread_rng());
        for (i, &x) in vec.iter().enumerate() {
            if x < subsample_amt {
                count[i] += 1;
            }
        }
    }
    count
}

#[inline]
fn _gcc_one_trial(
    total:u32
    , subsample_amt:u32
    , subsample_times:u32
    , n_times:u32
) -> u32 {
    let times:Array1<u32> = gcc_sample_pooled(total, subsample_amt, subsample_times);
    for x in times.into_iter() {
        if x < n_times {
            return 0
        }
    }
    1 // return 1
}

pub fn gcc_monte_carlo_prob_est(
    n_trials:usize
    , total:u32
    , subsample_amt:u32
    , subsample_times:u32
    , n_times:u32
) -> f32 {

    let success:u32 = (0..n_trials).into_par_iter()
        .map(|_| _gcc_one_trial(total, subsample_amt, subsample_times, n_times))
        .reduce(|| 0, |a,b| a + b);

    success as f32 / n_trials as f32
}
