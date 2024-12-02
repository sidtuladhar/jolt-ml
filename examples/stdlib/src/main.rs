use guest::{read_models, read_test_dataset, ModelInput};


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

    let model_input = ModelInput {
        test_features,
        actual_amounts,
        scaler,
        poly_ridge_model,
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
