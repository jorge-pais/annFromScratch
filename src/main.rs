mod matrix;
mod data_ingester;

use matrix::*;
use data_ingester::*;

use std::path::Path;

use raylib::prelude::{Color, RaylibDraw};
use raylib::consts::{TraceLogLevel, KeyboardKey};

fn render_matrix(images: Vec<Matrix>, labels: Matrix) {
// fn render_matrix() {
    let (mut rl, thread) = raylib::init()
        .size(800, 800)
        .title("Matrix")
        .log_level(TraceLogLevel::LOG_NONE)
        .build();

    const BLOCK_SIZE: i32 = 16;

    let mut index : usize = 0;

    while !rl.window_should_close() {
        if rl.is_key_pressed(KeyboardKey::KEY_SPACE){
            index += 1;
        }

        let mut d = rl.begin_drawing(&thread);

        d.clear_background(Color::BLACK);

        let m1 : &Matrix = &images[index];
        let label = labels.get(index, 0);

        for row in 0..m1.rows as i32 {
            for col in 0..m1.cols as i32{
                d.draw_rectangle(
                    col * BLOCK_SIZE, // x, these are swapped on the matrix
                    row * BLOCK_SIZE, // y
                    BLOCK_SIZE, // width
                    BLOCK_SIZE, // height
                    Color::WHITE.alpha(
                        // m1.get(row as usize, col as usize) / 255.0)
                        m1.get(row as usize, col as usize))
                    );
                // d.draw_text(&format!("({:02},{:02})", row, col), col*BLOCK_SIZE+BLOCK_SIZE/2, row*BLOCK_SIZE+BLOCK_SIZE/2, 1, Color::GREEN);
            }
        }
        d.draw_text(&format!("Index: {:}", index), 700, 380, 8, Color::WHITE);
        d.draw_text(&format!("Target: {:1.0}", label), 700, 400, 8, Color::WHITE);
    }
}

fn test_matrix() {
    let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let m2 = Matrix::new(2, 2, vec![4.0, 3.0, 2.0, 1.0]);

    println!("m1: {:?}", m1);       
    println!("m2: {:?}", m2);       
    println!("{:?}", add(&m1, &m2));
    println!("m1 dot m2: {:?}", dot(&m1, &m2));
    println!("3 * m1: {:?}", scale(&m1, 3.0));

    // sigmoid lambda wow
    let lambda = |x : f32| -> f32 { 1.0 / (1.0 + f32::exp(-x)) };
    println!("Apply sigmoid(m1): {:?}", matrix_apply(&m1, &lambda));
}

fn main() {
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
    let learning_rate : f32 = 0.10;

    let mut hidden_weights = Matrix::random(hidden, inputs);
    let mut output_weights = Matrix::random(output, hidden);

    for index in 0..picture.len() {
        let input_data = matrix_flatten(&picture[index]);
        let mut target_encoded = vec![0.0; 10]; 
        target_encoded[labels.get(index, 0) as usize] = 1.0; // hot one encoded

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
                -learning_rate 
            )
        );

        hidden_weights = add(&hidden_weights, 
            &scale(
                &dot(&multiply(&hidden_error, &matrix_apply(&hidden_out, &sigmoid_prime)),
                    &transpose(&input_data)
                ),
                -learning_rate 
            )
        );

        let mut error = 0.0;
        for i in 0..output_error.rows {
            error += output_error.get(i,0).powf(2.0);
        }

        error = error / 2.0;

        println!("Training step {:} error {:}", index, error);
    }

    // render_matrix(picture, labels);
}
