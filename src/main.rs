mod matrix;
mod data_ingester;
mod neural_network;

use matrix::*;
use data_ingester::*;

use std::path::Path;

use raylib::prelude::{Color, RaylibDraw};
use raylib::consts::{TraceLogLevel, KeyboardKey};

use crate::neural_network::NeuralNetwork;

fn render_matrix(images: Vec<Matrix>, labels: Vec<Matrix>) {
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
        let label = &labels[index];

        for row in 0..28 as i32 {
            println!("hey");
            for col in 0..28 as i32{
                d.draw_rectangle(
                    col * BLOCK_SIZE, // x, these are swapped on the matrix
                    row * BLOCK_SIZE, // y
                    BLOCK_SIZE, // width
                    BLOCK_SIZE, // height
                    Color::WHITE.alpha(
                        // m1.get(row as usize, col as usize) / 255.0)
                        m1.get(row as usize, col as usize)

                    )
                );

                println!("{:}", m1.get(row as usize, col as usize));
                // d.draw_text(&format!("({:02},{:02})", row, col), col*BLOCK_SIZE+BLOCK_SIZE/2, row*BLOCK_SIZE+BLOCK_SIZE/2, 1, Color::GREEN);
            }
        }
        d.draw_text(&format!("Index: {:}", index), 700, 380, 8, Color::WHITE);
        let mut num_label : usize = 0;
        for i in 0..label.rows{
            if label.get(i, 0) == 1.0 {
                num_label = i;
            }
        }
        d.draw_text(&format!("Target: {:1.0}", num_label), 700, 400, 8, Color::WHITE);
    }
}

fn main() {
    let train_images = read_images(Path::new("data/train-images-idx3-ubyte"));
    let train_labels = read_labels(Path::new("data/train-labels-idx1-ubyte"));

    assert_eq!(train_images.len(), train_labels.len(), "Training set and labels don't match");

    println!("Training data successfully loaded");

    let mut nn = NeuralNetwork::new(28*28, 200, 10);

    // nn.fit(train_images[0..1000].to_vec(), train_labels[0..1000].to_vec(), 0.01, 2);

    let test_images = read_images(Path::new("data/t10k-images-idx3-ubyte"));
    let test_labels = read_labels(Path::new("data/t10k-labels-idx1-ubyte"));

    render_matrix(train_images, train_labels);
}
