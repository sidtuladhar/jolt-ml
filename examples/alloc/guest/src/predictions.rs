// src/predictions.rs
extern crate alloc;
use alloc::vec::Vec;

use serde::{Serialize, Deserialize};

pub struct LinearRegressionParams {
    pub coefficients: Vec<f32>,
    pub intercept: f32// intercept is a single value but using Vec for consistency
}

pub struct RidgeRegressionParams {
    pub coefficients: Vec<f32>,
    pub intercept: f32
}

pub struct PolynomialRidgeRegressionParams {
    pub coefficients: Vec<f32>,
    pub intercept: f32,
}

pub struct ScalerParams {
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct Scaler {
    mean: Vec<f32>,
    scale: Vec<f32>,
}

impl Scaler {
    pub fn new(params: ScalerParams) -> Self {
        Scaler {
            mean: params.mean,
            scale: params.scale,
        }
    }

    pub fn transform(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&self.mean)
                    .zip(&self.scale)
                    .map(|((value, mean), scale)| (value - mean) / scale)
                    .collect()
            })
            .collect()
    }
}

#[derive(Serialize, Deserialize)]
pub struct LinearRegressionModel {
    coefficients: Vec<f32>,
    intercept: f32,
}

impl LinearRegressionModel {
    pub fn new(params: LinearRegressionParams) -> Self {
        LinearRegressionModel {
            coefficients: params.coefficients,
            intercept: params.intercept,
        }
    }

    pub fn predict(&self, x: &[Vec<f32>]) -> Vec<f32> {
        x.iter()
            .map(|row| {
                row.iter()
                    .zip(&self.coefficients)
                    .map(|(value, coef)| value * coef)
                    .sum::<f32>()
                    + self.intercept
            })
            .collect()
    }
}

#[derive(Serialize, Deserialize)]
pub struct RidgeRegressionModel {
    coefficients: Vec<f32>,
    intercept: f32,
}

impl RidgeRegressionModel {
    pub fn new(params: RidgeRegressionParams) -> Self {
        RidgeRegressionModel {
            coefficients: params.coefficients,
            intercept: params.intercept,
        }
    }

    pub fn predict(&self, x: &[Vec<f32>]) -> Vec<f32> {
        x.iter()
            .map(|row| {
                row.iter()
                    .zip(&self.coefficients)
                    .map(|(value, coef)| value * coef)
                    .sum::<f32>()
                    + self.intercept
            })
            .collect()
    }
}

// impl PolynomialRidgeRegressionModel {
//     pub fn new(params: PolynomialRidgeRegressionParams) -> Self {
//         PolynomialRidgeRegressionModel {
//             coefficients: Array1::from(params.coefficients),
//             intercept: params.intercept,
//             feature_names: params.feature_names,
//         }
//     }
//
//     pub fn predict(&self, x: &Array2<f32>) -> Array1<f32> {
//         // Assuming X has been preprocessed (scaled)
//         // Generate polynomial features manually
//         // For degree=2, include squares and pairwise products
//
//         let mut X_poly = Array2::<f32>::zeros((x.shape()[0], self.feature_names.len()));
//
//         for (i, row) in x.outer_iter().enumerate() {
//             for (j, feature_name) in self.feature_names.iter().enumerate() {
//                 // Simple parser for feature names like 'feature1', 'feature1^2', 'feature1 feature2'
//                 if feature_name.contains("^2") {
//                     let feature = feature_name.replace("^2", "");
//                     let idx = self.feature_names.iter().position(|f| f == &feature).unwrap();
//                     X_poly[[i, j]] = row[idx].powi(2);
//                 } else if feature_name.contains(' ') {
//                     let parts: Vec<&str> = feature_name.split(' ').collect();
//                     let idx1 = self.feature_names.iter().position(|f| f == parts[0]).unwrap();
//                     let idx2 = self.feature_names.iter().position(|f| f == parts[1]).unwrap();
//                     X_poly[[i, j]] = row[idx1] * row[idx2];
//                 } else {
//                     let idx = self.feature_names.iter().position(|f| f == feature_name).unwrap();
//                     X_poly[[i, j]] = row[idx];
//                 }
//             }
//         }
//
//         X_poly.dot(&self.coefficients) + self.intercept
//     }
// }
