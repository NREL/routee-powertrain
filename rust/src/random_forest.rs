use pyo3::types::PyType;
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;

use pyo3::prelude::*;

use anyhow::Result;

#[pyclass]
#[derive(Default)]
pub struct RustRandomForest {
    pub rf: Option<RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
}

#[pymethods]
impl RustRandomForest {
    #[new]
    pub fn new() -> RustRandomForest {
        RustRandomForest::default()
    }
    pub fn train(&mut self, train_data: Vec<Vec<f64>>, target: Vec<f64>) {
        let x = DenseMatrix::from_2d_vec(&train_data);

        let rf_params = RandomForestRegressorParameters::default()
            .with_max_depth(10)
            .with_min_samples_split(10)
            .with_n_trees(20)
            .with_seed(52);

        let rf = RandomForestRegressor::fit(&x, &target, rf_params).unwrap();
        self.rf = Some(rf);
    }

    pub fn predict(&self, test_data: Vec<Vec<f64>>) -> Result<Vec<f64>> {
        let x = DenseMatrix::from_2d_vec(&test_data);
        if let Some(rf) = &self.rf {
            let y = rf.predict(&x)?;
            Ok(y)
        } else {
            Err(anyhow::anyhow!("Random forest not trained"))
        }
    }

    pub fn to_json(&self) -> Result<String> {
        if let Some(rf) = &self.rf {
            let json = serde_json::to_string(&rf)?;
            Ok(json)
        } else {
            Err(anyhow::anyhow!("Random forest not trained"))
        }
    }

    #[classmethod]
    pub fn from_json(_cls: &PyType, json: String) -> Result<RustRandomForest> {
        let rf: RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            serde_json::from_str(&json)?;
        Ok(RustRandomForest { rf: Some(rf) })
    }

    pub fn to_bincode(&self) -> Result<Vec<u8>> {
        if let Some(rf) = &self.rf {
            let bincode = bincode::serialize(&rf)?;
            Ok(bincode)
        } else {
            Err(anyhow::anyhow!("Random forest not trained"))
        }
    }

    #[classmethod]
    pub fn from_bincode(_cls: &PyType, bincode: Vec<u8>) -> Result<RustRandomForest> {
        let rf: RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            bincode::deserialize(&bincode)?;
        Ok(RustRandomForest { rf: Some(rf) })
    }
}
