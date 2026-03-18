#[derive(Debug)]
pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize
}

impl Matrix{
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self{
        assert_eq!(rows*cols, data.len(), "data length must match matrix dimentions");
        Self{data, rows, cols} // when return no semicolon idiot
    }

    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
}

pub fn add(m1: &Matrix, m2: &Matrix) -> Matrix {
    assert!(m1.cols == m2.cols && m1.rows == m1.rows);

    let size = m1.rows * m1.cols;

    let mut result_data = vec![0.0; size]; // already initializes each element at 0

    for i in 0..m1.rows {
        for j in 0..m1.cols {
            let index = i * m1.cols + j;
            result_data[index] = m1.get(i, j) + m2.get(i, j);
        }
    }

    Matrix::new(m1.rows, m2.cols, result_data)
}

pub fn dot(m1: &Matrix, m2: &Matrix) -> Matrix{
    assert!(m1.cols == m2.rows, "Matrix dimentions are not compatible");

    let size = m1.rows * m2.cols;

    let mut result_data = vec![0.0; size];

    for i in 0..m1.rows {
        for j in 0..m2.cols {
            for k in 0..m1.cols {
                result_data[i * m1.rows + j] += m1.get(i, k) * m2.get(k, j);
            }
        }
    }

    Matrix::new(m1.rows, m2.cols, result_data)
}

pub fn multiply_scalar(m1: &Matrix, a: f32) -> Matrix {
    let size = m1.data.len();
    let mut result_data = vec![0.0; size];
    
    for i in 0..size {
        result_data[i] = m1.data[i] * a;
    }

    Matrix::new(m1.rows, m1.cols, result_data)
}

pub fn matrix_apply(m1: &Matrix, f: &dyn Fn(f32) -> f32) -> Matrix {
    let size = m1.data.len();
    let mut result_data = vec![0.0; size];
    
    for i in 0..size {
        result_data[i] = f(m1.data[i]);
    }

    Matrix::new(m1.rows, m1.cols, result_data)
}

