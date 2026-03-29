use rand::{RngExt};

#[derive(Debug)]
#[derive(Clone)]
pub struct Matrix {
    data: Vec<f32>,
    pub rows: usize,
    pub cols: usize
}

impl Matrix{

    pub fn new_empty(rows: usize, cols: usize) -> Self{
        let data = vec![0.0; rows*cols];
        Self{data, rows, cols} 
    }

    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self{
        assert_eq!(rows*cols, data.len(), "data length must match matrix dimensions");
        Self{data, rows, cols} 
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                rng.random_range(-0.5..=0.5) 
            })
            .collect();

        Self { data, rows, cols }
    }

    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    pub fn get_reshaped(&self, r: usize, c: usize, r_shape: usize, c_shape: usize) -> f32 {
        assert!(r_shape*c_shape == self.rows * self.cols, "Invalid reshape");
        self.data[r * c_shape + c]
    }
}

// add two matrices in-place
pub fn add_assign(m1: &mut Matrix, m2: &Matrix) {
    assert!(m1.cols == m2.cols && m1.rows == m2.rows, "Operand matrix dimensions do not match");

    for i in 0..m1.data.len() {
        m1.data[i] = m1.data[i] + m2.data[i];
    }
}

pub fn subtract_into(m1: &Matrix, m2: &Matrix, out: &mut Matrix) {
    assert!(m1.cols == m2.cols && m1.rows == m2.rows, "Operand matrix dimensions do not match");
    assert!(out.rows == m1.rows && out.cols == m2.cols, "Result matrix dimensions do not match");

    for i in 0..m1.data.len() {
        out.data[i] = m1.data[i] - m2.data[i];
    }
}

pub fn dot_into(m1: &Matrix, m2: &Matrix, out: &mut Matrix) {
    assert!(m1.cols == m2.rows,  "Operand matrix dimensions are not compatible");
    assert!(m1.rows == out.rows && m2.cols == out.cols, "Result matrix dimensions are not compatible");

    for x in out.data.iter_mut() {
        *x = 0.0;
    }

    // loop order is i-k-j such that we get better use of cache locality 
    // when accessing m2. I would the compiler sees this better than I do
    for i in 0..m1.rows {
        for k in 0..m1.cols {
            for j in 0..m2.cols {
                out.data[i * m2.cols + j] += m1.get(i, k) * m2.get(k, j);
            }
        }
    }
}

pub fn multiply_into(m1: &Matrix, m2: &Matrix, out: &mut Matrix) {
    assert!(m1.cols == m2.cols && m1.rows == m2.rows, "Operand matrix dimensions are not compatible");
    assert!(out.rows == m1.rows && out.cols == m2.cols, "Result matrix dimensions do not match");

    for i in 0..m1.rows {
        for j in 0..m1.cols {
            let index = i * m1.cols + j;
            out.data[index] = m1.data[index] * m2.data[index];
        }
    }
}

pub fn scale_assign(m1: &mut Matrix, a: f32) {
    for i in 0..m1.data.len() {
        m1.data[i] = m1.data[i] * a;
    }
}

pub fn matrix_apply_inplace<F: Fn(f32) -> f32>(m1: &mut Matrix, f: F) {
    for i in 0..m1.data.len() {
        m1.data[i] = f(m1.data[i]);
    }
}

pub fn transpose(m: &Matrix) -> Matrix {
    let mut result_data = vec![0.0; m.rows * m.cols];

    for i in 0..m.rows {
        for j in 0..m.cols {
            let old_index = i * m.cols + j;
            let new_index = j * m.rows + i;
            result_data[new_index] = m.data[old_index];
        }
    }

    Matrix::new(m.cols, m.rows, result_data)
}

pub fn matrix_flatten(m1: &Matrix) -> Matrix {
    Matrix::new(m1.rows * m1.cols, 1, m1.data.clone())
}

// this will apply a softmax to the result directly
pub fn softmax(m1: &mut Matrix) {
    let mut total : f32 = 0.0;

    for i in 0..m1.data.len() {
        total += f32::exp(m1.data[i]);
    }

    for i in 0..m1.data.len(){
        m1.data[i] = f32::exp(m1.data[i]) / total;
    }
}

pub fn argmax(m1: &Matrix) -> usize {
    let mut max_score : f32 = 0.0;
    let mut max_index : usize = 0;

    for i in 0..m1.data.len() {
        if m1.data[i] > max_score {
            max_score = m1.data[i];
            max_index = i;
        }
    }

    max_index
}

// fn test_matrix() {
//     let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
//     let m2 = Matrix::new(2, 2, vec![4.0, 3.0, 2.0, 1.0]);
//
//     println!("m1: {:?}", m1);       
//     println!("m2: {:?}", m2);       
//     println!("{:?}", add(&m1, &m2));
//     println!("m1 dot m2: {:?}", dot(&m1, &m2));
//     println!("3 * m1: {:?}", scale(&m1, 3.0));
//
//     // sigmoid lambda wow
//     let lambda = |x : f32| -> f32 { 1.0 / (1.0 + f32::exp(-x)) };
//     println!("Apply sigmoid(m1): {:?}", matrix_apply(&m1, &lambda));
// }
