pub mod random_forest;

use pyo3::prelude::*;

use crate::random_forest::RustRandomForest;

#[pymodule]
fn powertrain_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustRandomForest>()?;
    Ok(())
}
