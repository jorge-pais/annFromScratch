use crate::matrix::*;
use crate::data_ingester::*;

use std::path::Path;
use std::time::Instant;

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
    output_error : Matrix,  // shape outputs x 1

    buf1 : Matrix,
    buf2 : Matrix,
    buf3 : Matrix,
    buf4 : Matrix
}

impl NeuralNetwork {

pub fn new(inputs: usize, hidden: usize, outputs: usize) -> NeuralNetwork {
    let hidden_weights = Matrix::random(hidden, inputs);
    let output_weights = Matrix::random(outputs, hidden);
    let hidden_out = Matrix::new_empty(hidden, 1);
    let output_out = Matrix::new_empty(outputs, 1);
    let hidden_error = Matrix::new_empty(hidden, 1);
    let output_error = Matrix::new_empty(outputs, 1);

    let buf1 = Matrix::new_empty(outputs, 1);
    let buf2 = Matrix::new_empty(outputs, hidden);
    let buf3 = Matrix::new_empty(hidden, 1);
    let buf4 = Matrix::new_empty(hidden, inputs);

    Self{ 
        inputs, hidden, outputs, 
        hidden_weights, output_weights,
        hidden_out, output_out,
        hidden_error, output_error,
        buf1, buf2, buf3, buf4
    }
}

// activation function
fn sigmoid(x: f32) -> f32{
    1.0/(1.0 + f32::exp(-x))
}
// activation derivative
fn sigmoid_prime(x: f32) -> f32{
    x * (1.0 - x)
}

pub fn fit(&mut self, training_data: Vec<Matrix>, training_labels: Vec<Matrix>, learning_rate: f32, epochs: u16){
    assert_eq!(training_data[0].rows, self.inputs, "Training data is not the right shape");
    assert_eq!(training_labels[0].rows, self.outputs, "Training labels are not the right shape");

    assert_eq!(training_data.len(), training_labels.len(), "Training set and labels are not the same length");

    for epoch in 0..epochs {
        let now = Instant::now();
        println!("Running epoch {:}/{:}... ", epoch+1, epochs);

        for index in 0..training_data.len() {
            // forward propagation
            dot_into(&self.hidden_weights, &training_data[index], &mut self.hidden_out);
            matrix_apply_inplace(&mut self.hidden_out, &NeuralNetwork::sigmoid);

            dot_into(&self.output_weights, &self.hidden_out, &mut self.output_out);
            matrix_apply_inplace(&mut self.output_out, &NeuralNetwork::sigmoid);

            // find errors
            subtract_into(&training_labels[index], &self.output_out, &mut self.output_error);
            dot_into(&transpose(&self.output_weights), &self.output_error, &mut self.hidden_error);

            if index % 1000 == 0 {
                let mut mse = 0.0;
                for i in 0..self.output_error.rows {
                    mse += f32::powf(self.output_error.get(i, 0), 2.0);
                }
                mse /= self.output_error.rows as f32;
                println!("Epoch {:}/{:} image {:}/{:} - mse {:.6}", epoch + 1, epochs, index, training_data.len(), mse);
            }

            // backpropagation - output weights
            matrix_apply_inplace(&mut self.output_out, &NeuralNetwork::sigmoid_prime);
            multiply_into(&self.output_error, &self.output_out, &mut self.buf1);

            dot_into(&self.buf1, &transpose(&self.hidden_out), &mut self.buf2);
            scale_assign(&mut self.buf2, learning_rate);
            add_assign(&mut self.output_weights, &self.buf2);

            // backpropagation - hidden weights
            matrix_apply_inplace(&mut self.hidden_out, &NeuralNetwork::sigmoid_prime);
            multiply_into(&self.hidden_error, &self.hidden_out, &mut self.buf3);

            dot_into(&self.buf3, &transpose(&training_data[index]), &mut self.buf4);
            scale_assign(&mut self.buf4, learning_rate);
            add_assign(&mut self.hidden_weights, &self.buf4);
        }
        println!(" done! took {:} sec", now.elapsed().as_secs());
    }
}

pub fn predict(&self, m1: &Matrix) -> Matrix {

    let mut buf1 = Matrix::new_empty(self.hidden, 1);

    // forward propagation
    dot_into(&self.hidden_weights, &m1, &mut buf1);
    matrix_apply_inplace(&mut buf1, &NeuralNetwork::sigmoid);

    let mut buf2 = Matrix::new_empty(self.outputs, 1);

    dot_into(&self.output_weights, &buf1, &mut buf2);
    matrix_apply_inplace(&mut buf2, &NeuralNetwork::sigmoid);

    softmax(&mut buf2);

    buf2
}

} // impl NeuralNetwork

