// examples/fourier.rs
use one_indexed_vec::{VecIndexFromOne, vec1};
use std::f64::consts::PI;

/// Evaluates the Fourier series at point x
fn fourier_eval(x: f64, a0: f64, a: &VecIndexFromOne<f64>, b: &VecIndexFromOne<f64>) -> f64 {
    let mut result = a0;
    
    // 使用两个向量中较长的那个作为循环上限
    let max_len = a.len().max(b.len());
    for n in 1..=max_len {
        let nx = n as f64 * x;
        let a_n = if n <= a.len() { a[n] } else { 0.0 };
        let b_n = if n <= b.len() { b[n] } else { 0.0 };
        result += a_n * nx.cos() + b_n * nx.sin();
    }
    
    result
}

fn main() {
    // Square wave Fourier series coefficients (first 10 terms)
    let a0 = 0.0;
    let a = vec1![];
    let b = vec1![
        4.0 / PI,
        0.0,
        4.0 / (3.0 * PI),
        0.0,
        4.0 / (5.0 * PI),
        0.0,
        4.0 / (7.0 * PI),
        0.0,
        4.0 / (9.0 * PI),
        0.0,
    ];

    let value = fourier_eval(PI / 2.0, a0, &a, &b);
    println!("f(π/2) = {}", value);
    println!("Expected: ~1.0");
    println!("Error: {}", (value - 1.0).abs());
}