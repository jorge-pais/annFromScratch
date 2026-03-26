use crate::matrix::*;

use std::fs;
use std::path::Path;

pub fn read_images(path: &Path) -> Vec<Matrix> {

    let bytes = match fs::read(&path) {
        Err(why) => panic!("couldn't read {}: {}:", path.display(), why),
        Ok(file) => file,
    };

    // first two bytes are always zero
    // then this should be 0x08 specifying a unsigned byte
    // new bytes is the number of dimensions
    let magic_number = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic_number, 0x00000803u32);
    
    // this unwrap should be safe as the slice is always 4 bytes long...
    let num_images = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
    let rows = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
    let cols = u32::from_be_bytes(bytes[12..16].try_into().unwrap());

    println!("num_images: {:?}", num_images); // dimension 0
    println!("rows: {:?}", rows); // image dimension 1
    println!("cols: {:?}", cols); // image dimension 2

    let mut all_images : Vec<Matrix> = Vec::with_capacity(num_images as usize);

    let mut start : usize = 16;
    let mut end : usize;

    for _image in 0..num_images {
        let mut picture : Vec<f32> = Vec::with_capacity((rows * cols) as usize);

        end = start + 28*28;
        for byte in bytes[start..end].iter() {
            picture.push(*byte as f32 / 255.0);
        }

        all_images.push(Matrix::new((rows * cols) as usize, 1, picture));

        start = end;
    }
    all_images
}

pub fn read_labels(path: &Path) -> Vec<Matrix> {
    let bytes = match fs::read(&path) {
        Err(why) => panic!("couldn't read {}: {}:", path.display(), why),
        Ok(file) => file,
    };

    // first two bytes are always zero
    // then this should be 0x08 specifying a unsigned byte
    // new bytes is the number of dimensions
    let magic_number = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic_number, 0x00000801u32);

    let num_labels : usize = u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize;
    println!("num_labels: {:?}", num_labels);

    let mut labels : Vec<Matrix> = Vec::with_capacity(num_labels as usize);

    // this encodes as hot-one
    for byte in bytes[8..8+num_labels].iter() {
        let target = *byte as f32;

        let mut encoded = vec![0.0; 10];
        encoded[target as usize] = 1.0;
        
        labels.push(Matrix::new(10, 1, encoded));
    }
    
    labels
}
