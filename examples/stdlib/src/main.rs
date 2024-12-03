use guest::{ModelInput, models::TestData, models::LinearRegressionParams, models::RidgeRegressionParams, models::PolynomialRidgeRegressionParams, models::ScalerParams,
 predictions::Scaler, predictions::LinearRegressionModel, predictions::RidgeRegressionModel, predictions::PolynomialRidgeRegressionModel};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io;
use std::io::Read;
use ndarray::Array2;
use csv::ReaderBuilder;
use std::fmt;
use serde_json::from_str;
use std::env;

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

pub fn flatten(test_features: Vec<Vec<f32>>) -> Result<Array2<f32>, MyError> {
     
    let num_samples = test_features.len();
    let num_features = test_features[0].len();
    let flat_features: Vec<f32> = test_features.into_iter().flatten().collect();
    let x: Array2<f32> = Array2::from_shape_vec((num_samples, num_features), flat_features)
        .expect("Failed to create Array2 from shape");

   Ok(x) 
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

pub fn main() {
    let scaler_path = "./guest/model/scaler_params.json";
    let linear_model_path = "./guest/model/linear_regression_params.json";
    let ridge_model_path = "./guest/model/ridge_regression_params.json";
    let poly_ridge_model_path = "./guest/model/polynomial_ridge_regression_params.json";

    // Read the test dataset
    let Ok((test_features, actual_amounts)) = read_test_dataset() else { todo!() };
    // println!("test: {:?}", test_features);
    // Read the models
    let Ok((scaler, linear_model, ridge_model, poly_ridge_model)) =
    read_models(scaler_path, linear_model_path, ridge_model_path, poly_ridge_model_path) else {
        eprintln!("Error reading models: {:?}", read_models(scaler_path, linear_model_path, ridge_model_path, poly_ridge_model_path));
        return; // Exit or take alternative action
    };

    let Ok(x) = flatten(test_features) else { todo!() };
    let test = 3.32;

    let model_input = ModelInput {
        // test_features,
        // actual_amounts,
        test,
        scaler,
        ridge_model,
        x
    };
    println!("CREATING PROOF");
    // guest::load_model(model_input);
    let (test_prove, test_verify) = guest::build_load_model();
    println!("BUILT MODEL"); 
    let (test_output, test_proof) = test_prove(model_input);
    println!("PROVED MODEL");
    let test_is_valid = test_verify(test_proof);

    println!("model: {:?}", test_output);
    println!("model valid: {}", test_is_valid);

    // let (prove, verify) = guest::build_int_to_string();
    //
    // let (output, proof) = prove(81);
    // let is_valid = verify(proof);
    //
    // println!("int to string output: {:?}", output);
    // println!("int to string valid: {}", is_valid);
    //
    // let (prove, verify) = guest::build_string_concat();
    //
    // let (output, proof) = prove(20);
    // let is_valid = verify(proof);
    //
    // println!("string concat output: {:?}", output);
    // println!("string concat valid: {}", is_valid);
}
