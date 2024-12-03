#![cfg_attr(feature = "guest", no_std)]

use serde::{Deserialize, Serialize};
use ndarray::Array2;

extern crate alloc;
use alloc::vec::Vec;

pub mod models;
pub mod predictions;

use predictions::{Scaler, LinearRegressionModel, RidgeRegressionModel, PolynomialRidgeRegressionModel};

#[derive(Serialize, Deserialize)]
pub struct ModelInput {
    pub test: f32,
    pub scaler: Scaler,
    // pub poly_ridge_model: PolynomialRidgeRegressionModel,
    pub ridge_model: RidgeRegressionModel,
    pub x: Array2<f32>
}

#[jolt::provable(max_input_size = 100000000, max_output_size = 100000, memory_size = 18446000000000000000)] 
pub fn load_model(model_input: ModelInput) -> Vec<f32> {
    let X_scaled = model_input.scaler.transform(&model_input.x);
    
    let ridge_pred = model_input.ridge_model.predict(&X_scaled);
    // let poly_ridge_pred = model_input.poly_ridge_model.predict(&X_scaled);
    
    let combined_predictions = vec![ridge_pred.clone()];

    combined_predictions.into_iter().flat_map(|array| array.to_vec()).collect()
    // let mae_linear = compute_mae(&linear_pred, &actual_amounts);
    // let mse_linear = compute_mse(&linear_pred, &actual_amounts);
    // let rmse_linear = mse_linear.sqrt();
    // let r2_linear = compute_r2(&linear_pred, &actual_amounts);
    //
    // let mae_ridge = compute_mae(&ridge_pred, &actual_amounts);
    // let mse_ridge = compute_mse(&ridge_pred, &actual_amounts);
    // let rmse_ridge = mse_ridge.sqrt();
    // let r2_ridge = compute_r2(&ridge_pred, &actual_amounts);
    //
    // let mae_poly_ridge = compute_mae(&poly_ridge_pred, &actual_amounts);
    // let mse_poly_ridge = compute_mse(&poly_ridge_pred, &actual_amounts);
    // let rmse_poly_ridge = mse_poly_ridge.sqrt();
    // let r2_poly_ridge = compute_r2(&poly_ridge_pred, &actual_amounts);
    //
    // // Print the evaluation metrics
    // println!("----- Linear Regression Metrics -----");
    // println!("MAE: {}", mae_linear);
    // println!("MSE: {}", mse_linear);
    // println!("RMSE: {}", rmse_linear);
    // println!("R²: {}", r2_linear);
    //
    // println!("----- Ridge Regression Metrics -----");
    // println!("MAE: {}", mae_ridge);
    // println!("MSE: {}", mse_ridge);
    // println!("RMSE: {}", rmse_ridge);
    // println!("R²: {}", r2_ridge);
    //
    // println!("----- Polynomial Ridge Regression Metrics -----");
    // println!("MAE: {}", mae_poly_ridge);
    // println!("MSE: {}", mse_poly_ridge);
    // println!("RMSE: {}", rmse_poly_ridge);
    // println!("R²: {}", r2_poly_ridge);
    //
}

// fn compute_mae(predictions: &ndarray::Array1<f32>, actuals: &Vec<f32>) -> f32 {
//     let errors = predictions.iter().zip(actuals.iter()).map(|(p, a)| (p - a).abs());
//     errors.sum::<f32>() / predictions.len() as f32
// }
//
// fn compute_mse(predictions: &ndarray::Array1<f32>, actuals: &Vec<f32>) -> f32 {
//     let errors = predictions.iter().zip(actuals.iter()).map(|(p, a)| (p - a).powi(2));
//     errors.sum::<f32>() / predictions.len() as f32
// }
//
// fn compute_r2(predictions: &ndarray::Array1<f32>, actuals: &Vec<f32>) -> f32 {
//     let actual_mean = actuals.iter().sum::<f32>() / actuals.len() as f32;
//     let ss_tot: f32 = actuals.iter().map(|a| (*a - actual_mean).powi(2)).sum();
//     let ss_res: f32 = predictions.iter().zip(actuals.iter()).map(|(p, a)| (*a - p).powi(2)).sum();
//     1.0 - (ss_res / ss_tot)
// }



#[jolt::provable]
fn int_to_string(n: ModelInput) -> Vec<f32> {
    let mut res = Vec::<f32>::new(); 
    res[0] = n.test;
    res
}

#[jolt::provable]
fn string_concat(n: i32) -> String {
    let mut res = String::new();
    for i in 0..n {
        res += &i.to_string();
    }

    res
}
