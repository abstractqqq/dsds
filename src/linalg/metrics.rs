use ndarray::{Axis, Array2};
use ndarray::parallel::prelude::*;

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
        let norm:f64 = row.dot(&row).sqrt();
        row /= norm;
    });
}