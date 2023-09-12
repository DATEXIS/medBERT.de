use indicatif::ParallelProgressIterator;
use pyo3::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

#[pyfunction]
fn count_multi_chars(chars: Vec<&str>, string: &str) -> PyResult<Vec<usize>> {
    let count_vector = chars
        .par_iter()
        .progress_count(chars.len() as u64)
        .map(|char| {
            return string.matches(char).count();
        })
        .collect::<Vec<usize>>();

    Ok(count_vector)
}

#[pyfunction]
fn group_fast(input_ids: Vec<Vec<i32>>, max_len: i32) -> PyResult<Vec<Vec<i32>>> {
    let new_input_ids = input_ids
        .par_iter()
        .progress_count(input_ids.len() as u64)
        .map(|ids| {
            let mut new_ids: Vec<Vec<i32>> = Vec::new();
            for x in (0..ids.len()).step_by(max_len as usize) {
                new_ids.push(ids[x..(x + (max_len as usize))].to_vec())
            }
            return new_ids;
        })
        .flatten()
        .collect::<Vec<Vec<i32>>>();

    Ok(new_input_ids)
}

/// A Python module implemented in Rust.
#[pymodule]
fn char_counter(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_multi_chars, m)?)?;
    m.add_function(wrap_pyfunction!(group_fast, m)?)?;

    Ok(())
}
