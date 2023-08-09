use ndarray::{Axis, Array2};
use ndarray::parallel::prelude::*;
use rayon::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use polars::prelude::*;
use super::utils::split_offsets;

// Don't use these functions in Rust.. They shouldn't be in place operations
// from a user experience point of view. But they are copied from Python input
// and only serve this purpose. So I decided on these in place operations.


#[inline]
pub fn cosine_similarity(
    mut mat1:Array2<f64>,
    mut mat2:Array2<f64>,
    normalize:bool
) -> Array2<f64> {
    if normalize {
        row_normalize(&mut mat1);
        row_normalize(&mut mat2);
        mat1.dot(&mat2.t())
    } else {
        mat1.dot(&mat2.t())
    }
}

#[inline]
pub fn self_cosine_similarity(
    mut mat1:Array2<f64>,
    normalize:bool
) -> Array2<f64> {
    if normalize {
        row_normalize(&mut mat1);
        mat1.dot(&mat1.t())
    } else {
        mat1.dot(&mat1.t())
    }
}

#[inline]
fn row_normalize(mat:&mut Array2<f64>) {
    mat.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut row| {
        let norm: f64 = row.iter().fold(0., |acc, x| acc + x.powi(2)).sqrt();
        row /= norm;
    });
}

// 

pub enum InnerType {
    STRING,
    INTEGER
}

impl InnerType {

    pub fn from_str(s:&str) -> Option<InnerType> {
        match s {
            "str" => Some(InnerType::STRING),
            "int" => Some(InnerType::INTEGER),
            _ => None
        }
    }
}

fn compute_jaccard_similarity(sa: &Series, sb: &Series, st: &InnerType, include_null:bool) -> PolarsResult<Series> {
    let sa: &ChunkedArray<ListType> = sa.list()?;
    let sb: &ChunkedArray<ListType> = sb.list()?;

    let ca: ChunkedArray<Float64Type> = sa.into_iter().zip(sb.into_iter()).map(|(a, b)| {
        match (a, b) {
            (Some(a), Some(b)) => {
                let (mut s3_len, s1_len, s2_len) = match st {
                    InnerType::INTEGER => {
                        let (a, b) = (a.i64()?, b.i64()?);
                        let s1 = a.into_iter().collect::<PlHashSet<_>>();
                        let s2 = b.into_iter().collect::<PlHashSet<_>>();
                        (s1.intersection(&s2).count(), s1.len(), s2.len())
                    },
                    InnerType::STRING => {
                        let (a, b) = (a.utf8()?, b.utf8()?);
                        let s1 = a.into_iter().collect::<PlHashSet<_>>();
                        let s2 = b.into_iter().collect::<PlHashSet<_>>();
                        (s1.intersection(&s2).count(), s1.len(), s2.len())
                    }
                };
                // return similarity
                if (!include_null) & (a.null_count() > 0) & (b.null_count() > 0) {
                    s3_len -= 1;
                }
                Ok(Some(s3_len as f64 / (s1_len + s2_len - s3_len) as f64))
            },
            _ => Ok(None)
        }
    }).collect::<PolarsResult<Float64Chunked>>()?;
    Ok(ca.into_series())
}

pub fn list_jaccard_similarity(df:DataFrame, col_a: &str, col_b: &str, st: InnerType, include_null:bool) -> PolarsResult<DataFrame> {
    let offsets: Vec<(usize, usize)> = split_offsets(df.height(), rayon::current_num_threads());

    let dfs: Vec<DataFrame>= offsets.par_iter().map(|(offset, len)| {
        let sub_df = df.slice(*offset as i64, *len);
        let a: &Series = sub_df.column(col_a)?;
        let b: &Series = sub_df.column(col_b)?;
        let name = format!("{}_{}_jaccard", col_a, col_b);
        let out: Series = compute_jaccard_similarity(a, b, &st, include_null)?;
        df!(
            name.as_str() => out
        )
    }).collect::<PolarsResult<Vec<_>>>()?;
    accumulate_dataframes_vertical(dfs)
}

#[inline]
pub fn series_jaccard_similarity(
    a: Series, 
    b: Series, 
    st: InnerType, 
    include_null:bool,
    parallel: bool
) -> PolarsResult<f64> {

    // Jaccard similarity of two series. Stem only applies to string type.
    let na = a.null_count();
    let nb = b.null_count();
    let (mut s3_len, s1_len, s2_len) = match st {
        InnerType::INTEGER => {
            if parallel {
                let cas: [&ChunkedArray<Int64Type>; 2] = [a.i64()?, b.i64()?];
                let mut sets: Vec<PlHashSet<Option<i64>>> = Vec::with_capacity(2);
                cas.into_par_iter().map(|s| s.into_iter().collect::<PlHashSet<_>>())
                .collect_into_vec(&mut sets);
                let (s1, s2) = unsafe {
                    (sets.get_unchecked(0), sets.get_unchecked(1))
                };
                (s1.intersection(s2).count(), s1.len(), s2.len())
            } else {
                let (a, b) = (a.i64()?, b.i64()?);
                let s1 = a.into_iter().collect::<PlHashSet<_>>();
                let s2 = b.into_iter().collect::<PlHashSet<_>>();
                (s1.intersection(&s2).count(), s1.len(), s2.len())
            }
        },
        InnerType::STRING => {
            let (a, b) = (a.utf8()?, b.utf8()?);
            if parallel {
                let cas: [&ChunkedArray<Utf8Type>; 2] = [a, b];
                let mut sets: Vec<PlHashSet<Option<&str>>> = Vec::with_capacity(2);
                cas.into_par_iter().map(|s| s.into_iter().collect::<PlHashSet<_>>())
                .collect_into_vec(&mut sets);
                let (s1, s2) = unsafe {
                    (sets.get_unchecked(0), sets.get_unchecked(1))
                };
                (s1.intersection(s2).count(), s1.len(), s2.len())
            } else {
                let s1 = a.into_iter().collect::<PlHashSet<_>>();
                let s2 = b.into_iter().collect::<PlHashSet<_>>();
                (s1.intersection(&s2).count(), s1.len(), s2.len())
            }
        }
    };
    if (!include_null) & (na > 0) & (nb > 0) {
        s3_len -= 1;
    }
    Ok(s3_len as f64 / (s1_len + s2_len - s3_len) as f64)
}

