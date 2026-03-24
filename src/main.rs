mod matrix;
mod data_ingester;

use matrix::*;
use data_ingester::*;

use raylib::prelude::{Color, RaylibDraw};
use raylib::consts::{TraceLogLevel, KeyboardKey};

fn render_matrix(images: Vec<Matrix>) {
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

        for row in 0..m1.rows as i32 {
            for col in 0..m1.cols as i32{
                d.draw_rectangle(
                    col * BLOCK_SIZE, // x, these are swapped on the matrix
                    row * BLOCK_SIZE, // y
                    BLOCK_SIZE, // width
                    BLOCK_SIZE, // height
                    Color::WHITE.alpha(
                        m1.get(row as usize, col as usize) / 255.0)
                    );
                // d.draw_text(&format!("({:02},{:02})", row, col), col*BLOCK_SIZE+BLOCK_SIZE/2, row*BLOCK_SIZE+BLOCK_SIZE/2, 1, Color::GREEN);
            }
        }
    }
}

fn main() {
    let m1 = Matrix::new(2, 2, vec![
        1.0, 2.0, 
        3.0, 4.0]);
    let m2 = Matrix::new(2, 2, vec![
        4.0, 3.0, 
        2.0, 1.0]);

    println!("m1: {:?}", m1);       
    println!("m2: {:?}", m2);       
    println!("{:?}", add(&m1, &m2));
    println!("m1 dot m2: {:?}", dot(&m1, &m2));
    println!("3 * m1: {:?}", multiply_scalar(&m1, 3.0));

    // sigmoid lambda wow
    let lambda = |x : f32| -> f32 { 1.0 / (1.0 + f32::exp(-x)) };
    println!("Apply sigmoid(m1): {:?}", matrix_apply(&m1, &lambda));

    let picture = read_image();
    render_matrix(picture);
}
