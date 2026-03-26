use crate::matrix::*;
use crate::data_ingester::*;

use std::path::Path;

#[derive(Debug)]
struct MultilayerPerceptron {
    inputs: usize,
    hidden: usize,
    outputs: usize,
}

fn test_nn(){
    let picture = read_images(Path::new("data/train-images-idx3-ubyte"));
    let labels = read_labels(Path::new("data/train-labels-idx1-ubyte"));

    assert_eq!(picture.len(), labels.rows, "Training set and labels don't match");

    // sigmoid activation function and the derivative
    let sigmoid = |x : f32| -> f32 { 1.0 / (1.0 + f32::exp(-x)) };
    let sigmoid_prime = |x : f32| -> f32 { x * (1.0 - x) };

    // network parameters 
    let inputs : usize = 28*28;
    let hidden : usize = 200;
    let output : usize = 10;
    let learning_rate : f32 = 0.01;

    let mut hidden_weights = Matrix::random(hidden, inputs);
    let mut output_weights = Matrix::random(output, hidden);

    for index in 0..picture.len() {
        let input_data = matrix_flatten(&picture[index]);
        let mut target_encoded = vec![0.01; 10]; 
        target_encoded[labels.get(index, 0) as usize] = 0.99; // hot one encoded

        // forward propagation
        let hidden_out = matrix_apply(&dot(&hidden_weights, &input_data), &sigmoid);
        let output_out = matrix_apply(&dot(&output_weights, &hidden_out), &sigmoid);

        // find errors
        let output_error = subtract(&Matrix::new(10, 1, target_encoded), &output_out);
        let hidden_error = dot(&transpose(&output_weights), &output_error);

        // backpropagation
        output_weights = add(&output_weights, 
            &scale(
                &dot(&multiply(&output_error, &matrix_apply(&output_out, &sigmoid_prime)),
                    &transpose(&hidden_out)
                ),
                learning_rate 
            )
        );

        hidden_weights = add(&hidden_weights, 
            &scale(
                &dot(&multiply(&hidden_error, &matrix_apply(&hidden_out, &sigmoid_prime)),
                    &transpose(&input_data)
                ),
                learning_rate 
            )
        );

        // let mut error = 0.0;
        // for i in 0..output_error.rows {
        //     error += output_error.get(i,0).powf(2.0);
        // }
        // error = error / 2.0;
        // println!("Training step {:} error {:}", index, error);
    }
}
