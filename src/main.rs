mod matrix;
use matrix::*;

use std::fs;
// use std::fs::File;
use std::path::Path;

fn read_image() -> Matrix {
    let root = Path::new("data/");
    let image = root.join("t10k-images-idx3-ubyte");
    
    let bytes = match fs::read(&image) {
        Err(why) => panic!("couldn't read {}: {}:", image.display(), why),
        Ok(file) => file,
    };

    // first two bytes are always zero
    // then this should be 0x08 specifying a unsigned byte
    // new bytes is the number of dimensions. in this file it is stored as a  
    let magic_number = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic_number, 0x00000803u32);
    
    // this unwrap should be safe as the slice is always 4 bytes long...
    let num_images = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
    let rows = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
    let cols = u32::from_be_bytes(bytes[12..16].try_into().unwrap());

    println!("num_images: {:?}", num_images); // image index
    println!("rows: {:?}", rows); // image dimension 1
    println!("cols: {:?}", cols); // image dimension 2

    for byte in bytes[4..733].iter() {
        print!("0x{:02X} ", byte);
    }
    println!("");
    
    let picture = Vec::<f32>::from(bytes[16..745]);

    Matrix::new(27, 27, picture)
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

    read_image();
}
