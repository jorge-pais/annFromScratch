use crate::matrix::*;
use crate::data_ingester::*;

use std::path::Path;

// represents a single hidden layer neural network
#[derive(Debug)]
pub struct NeuralNetwork {
    inputs: usize,
    hidden: usize,
    outputs: usize,
    hidden_weights : Matrix, // shape hidden x inputs
    output_weights : Matrix, // shape output x hidden

    hidden_out : Matrix, // shape hidden x 1
    output_out : Matrix, // shape output x 1
    hidden_error : Matrix, // shape hidden x 1
    output_error : Matrix  // shape outputs x 1
}

impl NeuralNetwork {

pub fn new(inputs: usize, hidden: usize, outputs: usize) -> NeuralNetwork {
    let hidden_weights = Matrix::random(hidden, inputs);
    let output_weights = Matrix::random(outputs, hidden);
    let hidden_out = Matrix::new_empty(hidden, 1);
    let output_out = Matrix::new_empty(outputs, 1);
    let hidden_error = Matrix::new_empty(hidden, 1);
    let output_error = Matrix::new_empty(outputs, 1);

    Self{ 
        inputs, hidden, outputs, 
        hidden_weights, output_weights,
        hidden_out, output_out,
        hidden_error, output_error
    }
}

// activation function
fn sigmoid(x: f32) -> f32{
    1.0/(1.0/f32::exp(-x))
}
// activation derivative
fn sigmoid_prime(x: f32) -> f32{
    x * (1.0 - x)
}

pub fn fit(&mut self, training_data: Vec<Matrix>, training_labels: Vec<Matrix>, learning_rate: f32, epochs: u16){
    assert_eq!(training_data[0].rows, self.inputs, "Training data is not the right shape");
    assert_eq!(training_labels[0].rows, self.outputs, "Training labels are not the right shape");

    assert_eq!(training_data.len(), training_labels.len(), "Training set and labels are not the same length");

    for _epoch in 0..epochs {
        for index in 0..training_data.len() {
            // forward propagation
            dot_into(&self.hidden_weights, &training_data[index], &mut self.hidden_out);
            matrix_apply_inplace(&mut self.hidden_out, &NeuralNetwork::sigmoid);

            dot_into(&self.output_weights, &self.hidden_out, &mut self.output_out);
            matrix_apply_inplace(&mut self.hidden_out, &NeuralNetwork::sigmoid);

            // find errors
            subtract_into(&training_labels[index], &self.output_out, &mut self.output_error);
            dot_into(&transpose(&self.output_weights), &self.output_error, &mut self.hidden_error);

            // backpropagation - output weights
            matrix_apply_inplace(&mut self.output_out, &NeuralNetwork::sigmoid_prime);
            let mut buf1 = Matrix::new_empty(self.output_error.rows, self.output_error.cols);
            multiply_into(&self.output_error, &self.output_out, &mut buf1);

            let mut buf2 = Matrix::new_empty(self.output_weights.rows, self.output_weights.cols);
            dot_into(&buf1, &transpose(&self.hidden_out), &mut buf2);
            scale_assign(&mut buf2, learning_rate);
            add_assign(&mut self.output_weights, &buf2);

            // backpropagation - hidden weights
            matrix_apply_inplace(&mut self.hidden_out, &NeuralNetwork::sigmoid_prime);
            let mut buf3 = Matrix::new_empty(self.hidden_error.rows, self.hidden_error.cols);
            multiply_into(&self.hidden_error, &self.hidden_out, &mut buf3);

            let mut buf4 = Matrix::new_empty(self.hidden_weights.rows, self.hidden_weights.cols);
            dot_into(&buf3, &transpose(&training_data[index]), &mut buf4);
            scale_assign(&mut buf4, learning_rate);
            add_assign(&mut self.hidden_weights, &buf4);
        }
    }
}

} // impl NeuralNetwork

