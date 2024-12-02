use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io;
use std::io::Read;
use ndarray::Array2;
use csv::ReaderBuilder;
use std::fmt;
use serde_json::from_str;
use std::env;

mod models;
mod predictions;

use models::{TestData, LinearRegressionParams, RidgeRegressionParams, PolynomialRidgeRegressionParams, ScalerParams};
use predictions::{Scaler, LinearRegressionModel, RidgeRegressionModel, PolynomialRidgeRegressionModel};

#[derive(Serialize, Deserialize)]
pub struct ModelInput {
    pub test_features: Vec<Vec<f32>>,
    pub actual_amounts: Vec<f32>,
    pub scaler: Scaler,
    pub poly_ridge_model: PolynomialRidgeRegressionModel
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MyError {
    FileNotFound(String),
    ParseError(String),
    IoError(String),
    InvalidInput(String)
}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MyError::FileNotFound(msg) => write!(f, "File not found: {}", msg),
            MyError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            MyError::IoError(msg) => write!(f, "IO error: {}", msg),
            MyError::InvalidInput(msg) => write!(f, "Invlaid Input: {}", msg)
        }
    }
}

impl std::error::Error for MyError {}

impl From<io::Error> for MyError {
    fn from(error: io::Error) -> Self {
        MyError::IoError(error.to_string())
    }
}

impl From<serde_json::Error> for MyError {
    fn from(error: serde_json::Error) -> Self {
        MyError::ParseError(error.to_string())
    }
}

#[jolt::provable(max_input_size = 100000, max_output_size = 100000, stack_size = 1844600000000000000, memory_size = 18446000000000000000)] pub fn load_model(model_input: ModelInput) -> Result<Vec<f32>, MyError> {
    
    if model_input.test_features.is_empty() {
        return Err(MyError::InvalidInput("Test features vector is empty.".into()));
    }

    let num_samples = model_input.test_features.len();
    let num_features = model_input.test_features[0].len();
    let flat_features: Vec<f32> = model_input.test_features.into_iter().flatten().collect();
    let x: Array2<f32> = Array2::from_shape_vec((num_samples, num_features), flat_features)
        .expect("Failed to create Array2 from shape");


    let X_scaled = model_input.scaler.transform(&x);
    
    // Make predictions
    // let linear_pred = model_input.linear_model.predict(&X_scaled);
    // let ridge_pred = model_input.ridge_model.predict(&X_scaled);
    let poly_ridge_pred = model_input.poly_ridge_model.predict(&X_scaled);
    
    let combined_predictions = vec![poly_ridge_pred.clone()];

    // println!("Linear Regression Prediction: {}", linear_pred[0]);
    // println!("Ridge Regression Prediction: {}", ridge_pred[0]);
    // println!("Polynomial Ridge Regression Prediction: {}", poly_ridge_pred[0]);
    //

    Ok(combined_predictions.into_iter().flat_map(|array| array.to_vec()).collect())
    // // Compute evaluation metrics
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

fn compute_mae(predictions: &ndarray::Array1<f32>, actuals: &Vec<f32>) -> f32 {
    let errors = predictions.iter().zip(actuals.iter()).map(|(p, a)| (p - a).abs());
    errors.sum::<f32>() / predictions.len() as f32
}

fn compute_mse(predictions: &ndarray::Array1<f32>, actuals: &Vec<f32>) -> f32 {
    let errors = predictions.iter().zip(actuals.iter()).map(|(p, a)| (p - a).powi(2));
    errors.sum::<f32>() / predictions.len() as f32
}

fn compute_r2(predictions: &ndarray::Array1<f32>, actuals: &Vec<f32>) -> f32 {
    let actual_mean = actuals.iter().sum::<f32>() / actuals.len() as f32;
    let ss_tot: f32 = actuals.iter().map(|a| (*a - actual_mean).powi(2)).sum();
    let ss_res: f32 = predictions.iter().zip(actuals.iter()).map(|(p, a)| (*a - p).powi(2)).sum();
    1.0 - (ss_res / ss_tot)
}

pub fn read_test_dataset() -> Result<(Vec<Vec<f32>>, Vec<f32>), MyError> {
    match env::current_dir() {
            Ok(path) => println!("Current working directory: {}", path.display()),
            Err(e) => eprintln!("Error getting current directory: {}", e),
    }
    let mut file = File::open("./guest/model/Test_Dataset.csv").expect("Test_Dataset.csv not found.");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read Test_Dataset.csv");

    // Parse the CSV into a vector of TestData
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(contents.as_bytes());

    let mut test_features = Vec::new();
    let mut actual_amounts = Vec::new();

    for result in rdr.deserialize() {
        let record: TestData = result.expect("Failed to deserialize record.");
        // Collect features into a vector (excluding 'amount')
        test_features.push(vec![
            record.quantity,
            record.price,
            record.discount_applied,
            record.IsAL,
            record.IsAK,
            record.IsAZ,
            record.IsAR,
            record.IsCA,
            record.IsCO,
            record.IsCT,
            record.IsDE,
            record.IsFL,
            record.IsGA,
            record.IsHI,
            record.IsID,
            record.IsIL,
            record.IsIN,
            record.IsIA,
            record.IsKS,
            record.IsKY,
            record.IsLA,
            record.IsME,
            record.IsMD,
            record.IsMA,
            record.IsMI,
            record.IsMN,
            record.IsMS,
            record.IsMO,
            record.IsMT,
            record.IsNE,
            record.IsNV,
            record.IsNH,
            record.IsNJ,
            record.IsNM,
            record.IsNY,
            record.IsNC,
            record.IsND,
            record.IsOH,
            record.IsOK,
            record.IsOR,
            record.IsPA,
            record.IsRI,
            record.IsSC,
            record.IsSD,
            record.IsTN,
            record.IsTX,
            record.IsUT,
            record.IsVT,
            record.IsVA,
            record.IsWA,
            record.IsWV,
            record.IsWI,
            record.IsWY,
            record.IsCash,
            record.IsPayPal,
            record.IsDebitCard,
            record.IsCreditCard,
            record.IsBooks,
            record.IsHomeDecor,
            record.IsElectronics,
            record.IsClothing,
        ]);
        actual_amounts.push(record.amount);
    }
    
    Ok((test_features, actual_amounts))
}

pub fn read_models(
    scaler_path: &str,
    linear_model_path: &str,
    ridge_model_path: &str,
    poly_ridge_model_path: &str,
) -> Result<(Scaler, LinearRegressionModel, RidgeRegressionModel, PolynomialRidgeRegressionModel), MyError> {
    match env::current_dir() {
            Ok(path) => println!("Current working directory: {}", path.display()),
            Err(e) => eprintln!("Error getting current directory: {}", e),
    }
    // Read and deserialize Scaler

    let mut scaler_file = File::open(scaler_path)?;
    let mut scaler_contents = String::new();
    scaler_file.read_to_string(&mut scaler_contents)?;
    let scalar_params: ScalerParams = from_str(&scaler_contents)?;
    let scaler = Scaler::new(scalar_params);

    // Read and deserialize LinearRegressionModel
    let mut linear_file = File::open(linear_model_path)?;
    let mut linear_contents = String::new();
    linear_file.read_to_string(&mut linear_contents)?;
    let linear_params: LinearRegressionParams =  from_str(&linear_contents)?;
    let linear_model = LinearRegressionModel::new(linear_params);

    // Read and deserialize RidgeRegressionModel
    let mut ridge_file = File::open(ridge_model_path)?;
    let mut ridge_contents = String::new();
    ridge_file.read_to_string(&mut ridge_contents)?;
    let ridge_params: RidgeRegressionParams = from_str(&ridge_contents)?;
    let ridge_model = RidgeRegressionModel::new(ridge_params);

    // Read and deserialize PolynomialRidgeRegressionModel
    let mut poly_ridge_file = File::open(poly_ridge_model_path)?;
    let mut poly_ridge_contents = String::new();
    poly_ridge_file.read_to_string(&mut poly_ridge_contents)?;
    let poly_ridge_params: PolynomialRidgeRegressionParams =  from_str(&poly_ridge_contents)?;
    let poly_ridge_model = PolynomialRidgeRegressionModel::new(poly_ridge_params);

    Ok((scaler, linear_model, ridge_model, poly_ridge_model))
}



#[jolt::provable]
fn int_to_string(n: i32) -> String {
     
    let mut file = File::open("./guest/model/Test_Dataset.csv").expect("Test_Dataset.csv not found.");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read Test_Dataset.csv");

    n.to_string()
}

#[jolt::provable]
fn string_concat(n: i32) -> String {
    let mut res = String::new();
    for i in 0..n {
        res += &i.to_string();
    }

    res
}
