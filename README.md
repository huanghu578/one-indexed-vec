# one-indexed-vec

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70+-blue.svg)](https://www.rust-lang.org/)
[![no_std](https://img.shields.io/badge/no__std-compatible-green.svg)](https://docs.rs/one-indexed-vec)

**one-indexed-vec** is a Rust library that provides a 1-indexed wrapper around `Vec<T>`, aligning with the mathematical conventions of Fortran, MATLAB, spreadsheets, and countless formulas.

## 📌 Philosophy

Mathematical notation naturally indexes vectors from 1: `x₁, x₂, ..., xₙ`. This crate lets you **transcribe formulas directly into code** without the mental overhead of subtracting 1 at every access.

✨ Features  

✅ 1-based indexing: v[1] accesses the first element, v[n] accesses the nth  
✅ Type-safe indexing: Index1 strong type prevents 0-based/index confusion  
✅ Zero-cost abstraction: Wraps std::Vec with no runtime overhead  
✅ Rich numeric algorithms: Prefix sums, sliding windows, finite differences, moving averages, and more  
✅ no_std support: Disable std feature for embedded environments (requires alloc)  
✅ Safe indexing: get(0) returns None, and internal conversions use checked_sub  

🚀 Quick Start  

Installation  
Add to your Cargo.toml:

[dependencies]  
one-indexed-vec = "0.1.2"  


Basic Usage  
use one_indexed_vec::{VecIndexFromOne, vec1};  
// Create a 1-indexed vector from a Vec  
let mut v = VecIndexFromOne::from(vec![10, 20, 30]);  
v.push(40);  
// Indexing starts at 1  
assert_eq!(v[1], 10);  
assert_eq!(v[3], 30);  
assert_eq!(v[4], 40);  
// Index 0 is invalid  
assert_eq!(v.get(0), None);  
assert_eq!(v.first(), Some(&10));  

// `vec1!` macro for convenient initialization  
let v2 = vec1![1, 2, 3];  
 
// Dot product  
let dpu = v.dot_product_upto(&v2, 3); // 10*1 + 20*2 + 30*3 = 140  
assert_eq!(dpu, 140);  

📚 Detailed Features  

1. Initialization: vec1! Macro  

use one_indexed_vec::{VecIndexFromOne, vec1};  

let v = vec1![1, 2, 3, 4, 5];     
let h4: VecIndexFromOne<i32>=vec1![]; // create a empty VecIndexFromOne  

// From existing Vec  
let raw = vec![10, 20, 30];  
let v = VecIndexFromOne::from(raw);  
assert_eq!(v[1], 10);  

// With capacity  
let v: VecIndexFromOne<i32> = VecIndexFromOne::with_capacity(100);  

2. Index 0 Returns None  

The get(0) method explicitly returns None, reinforcing the 1-based semantic:  

use one_indexed_vec::vec1;  

let v = vec1![100, 200, 300];  
// get(0) always returns None  
assert_eq!(v.get(0), None);  

// Valid indices return Some(&element)  
assert_eq!(v.get(1), Some(&100));  
assert_eq!(v.get(3), Some(&300));  
assert_eq!(v.get(4), None);  // Out of bounds  

// Indexing with [] panics on 0  
// let x = v[0];  // ❌ Panics with range error  

3. Get Length: .len()  

use one_indexed_vec::vec1;  

let v = vec1![10, 20, 30, 40, 50];  
assert_eq!(v.len(), 5);  

// len() is always ≥ 1  
let single = vec1![42];  
assert_eq!(single.len(), 1);  

4. Reverse: .reverse() / .reversed()  
Mutably reverse in-place:  

use one_indexed_vec::vec1;  

let mut v = vec1![1, 2, 3, 4, 5];  
v.reverse();  
// Indices remain 1-based after reversal  
assert_eq!(v[1], 5);  
assert_eq!(v[2], 4);  
assert_eq!(v[3], 3);  
assert_eq!(v[4], 2);  
assert_eq!(v[5], 1);  


Create a reversed copy without modifying the original:  


let v = vec1![1, 2, 3];  
let reversed = v.reversed();  // v is still [1, 2, 3]  
assert_eq!(reversed, vec1![3, 2, 1]);  

5. Product: .product()  
Compute the product of all elements (for numeric types):  

use one_indexed_vec::vec1;  
let v = vec1![2, 3, 4];  
let prod: i32 = v.product();  
assert_eq!(prod, 2 * 3 * 4);  // 24  

let v_float = vec1![1.5, 2.0, 4.0];  
let prod_float: f64 = v_float.product();  
assert_eq!(prod_float, 12.0);  
Also works with iterators:  
let v = vec1![2, 3, 4];  
let prod: i32 = v.iter().product();  
assert_eq!(prod, 24);  

6. Prefix Sum: .prefix_sum_view()  
Compute cumulative sums with O(1) range queries:  

use one_indexed_vec::vec1;  
let v = vec1![1, 2, 3, 4, 5];  
let prefix = v.prefix_sum_view();  

// Prefix sums: [1, 3, 6, 10, 15]  

// Sum of elements from index 2 to 4: 2 + 3 + 4 = 9  
assert_eq!(prefix.range_sum(2, 4), 9);  

// Total sum of all elements  
assert_eq!(prefix.total_sum(), 15);  
Use case: Efficient subarray sum queries, integral approximation.  

7. Dot Product: .dot_product()  
Calculate the dot product of two vectors:  


use one_indexed_vec::vec1;  

let a = vec1![1, 2, 3];  
let b = vec1![4, 5, 6];  

let dot = a.dot_product(&b);  
assert_eq!(dot, 1*4 + 2*5 + 3*6);  // 32  

With a limited number of terms:  

let a = vec1![1, 2, 3, 4];  
let b = vec1![10, 20, 30];  

// Dot product of first 3 elements only  
let dpu = a.dot_product_upto(&b, 3);  // 1*10 + 2*20 + 3*30 = 140  
assert_eq!(dpu, 140);  

8. Windows: .windows(), .windows_mut()  
Zero-copy sliding windows for filtering and convolution:  

use one_indexed_vec::vec1;  

let v = vec1![1, 2, 3, 4, 5, 6];  

// Immutable windows  
let mut windows = v.windows(3);  
assert_eq!(windows.next().unwrap(), &[1, 2, 3]);  
assert_eq!(windows.next().unwrap(), &[2, 3, 4]);  
assert_eq!(windows.next().unwrap(), &[3, 4, 5]);  
assert_eq!(windows.next().unwrap(), &[4, 5, 6]);  
assert!(windows.next().is_none());  
Mutable windows for in-place modification:  


let mut v = vec1![1, 2, 3, 4, 5];  

// Double all elements using window size 1  
for window in v.windows_mut(1) {  
    window[0] *= 2;  
}  
assert_eq!(v.as_slice(), &[2, 4, 6, 8, 10]);  

// Modify a specific window (indices 2 to 4)  
{  
    let mut window = v.window_mut(2, 4);  
    for i in 1..=window.len() {  
        window[i] += 1;  
    }  
}  
assert_eq!(v.as_slice(), &[2, 5, 7, 9, 10]);  
Use case: Moving averages, local filters, edge detection.  

9. Finite Differences  
Compute derivatives of discretized functions:  

use one_indexed_vec::vec1;  
let v = vec1![1, 4, 9, 16, 25];  // f(x) = x²  
let diff = v.finite_difference();  
assert_eq!(diff.as_slice(), &[3, 5, 7, 9]);  // ≈ 2x+1  

let diff2 = v.second_finite_difference();  
assert_eq!(diff2.as_slice(), &[2, 2, 2]);    // second derivative is constant  

10. Moving Average (Smoothing)  

use one_indexed_vec::vec1;  

let v = vec1![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];  
let smoothed = v.moving_average(3);  
assert_eq!(smoothed.as_slice(), &[2.0, 3.0, 4.0, 5.0, 6.0]);  

11. Fourier Series Evaluation  
The 1-indexed vector is especially elegant for Fourier series, where coefficients are naturally indexed from 1:  


use one_indexed_vec::{VecIndexFromOne, vec1};  
use std::f64::consts::PI;  

/// Evaluates the Fourier series at point x  
/// 
/// a0: constant term (a₀/2)  
/// a: cosine coefficient vector (a₁, a₂, a₃, ...)  
/// b: sine coefficient vector (b₁, b₂, b₃, ...)  
fn fourier_eval(x: f64, a0: f64, a: &VecIndexFromOne<f64>, b: &VecIndexFromOne<f64>) -> f64 {  
    let max_len = a.len().max(b.len());  
    let mut result = a0;  
    
    for n in 1..=max_len {  
        let nx = n as f64 * x;  
        let a_n = if n <= a.len() { a[n] } else { 0.0 };  
        let b_n = if n <= b.len() { b[n] } else { 0.0 };  
        result += a_n * nx.cos() + b_n * nx.sin();  
    }  
    
    result  
}  

// Square wave Fourier series (first 10 terms)  
let a0 = 0.0;                                  // DC component  
let a = vec1![];                               // Square wave has no cosine terms  
let b = vec1![  
    4.0 / PI,                                  // b₁  
    0.0,                                       // b₂  
    4.0 / (3.0 * PI),                          // b₃  
    0.0,                                       // b₄  
    4.0 / (5.0 * PI),                          // b₅  
    0.0,                                       // b₆  
    4.0 / (7.0 * PI),                          // b₇  
    0.0,                                       // b₈  
    4.0 / (9.0 * PI),                          // b₉  
    0.0,                                       // b₁₀  
];  

// Evaluate at x = π/2 (should approximate 1.0)  
let value = fourier_eval(PI / 2.0, a0, &a, &b);  
println!("f(π/2) = {}", value);  

12. Capacity Management  
Optimize memory usage:  

use one_indexed_vec::{vec1, VecIndexFromOne};  

let mut v = VecIndexFromOne::with_capacity(1000);  
v.extend(vec1![1, 2, 3, 4, 5].iter().copied());  

// Shrink if capacity > 4 × length  
v.shrink_to_fit_if(4);  
assert!(v.capacity() < 20);  

// Or shrink if excess capacity exceeds a threshold  
let mut v2 = VecIndexFromOne::with_capacity(1000);  
v2.extend(vec1![1, 2, 3].iter().copied());  
v2.shrink_to_fit_if_excess(10);  
assert!(v2.capacity() < 100);  

13. Type-Safe Indexing with Index1  
When you want compile-time guarantees that only 1-based indices are used:  

use one_indexed_vec::{VecIndexFromOne, Index1, vec1};  

let v = vec1![10, 20, 30];  
let first = Index1::new(1);   // Explicit 1-based position  
assert_eq!(v[first], 10);  

// Cannot accidentally pass a 0-based index  
// let invalid = Index1::new(0);  // ❌ Panics: index must be ≥ 1  

🔧 Complete Example  
use one_indexed_vec::{VecIndexFromOne, vec1};  

fn main() {  
    // 1. Initialize  
    let data = vec1![2, 4, 6, 8, 10];  
    
    // 2. Access (1-based)  
    assert_eq!(data[1], 2);  
    assert_eq!(data[3], 6);  
    assert_eq!(data[5], 10);  
    assert_eq!(data.get(0), None);  
    
    // 3. Length  
    assert_eq!(data.len(), 5);  
    
    // 4. Reverse  
    let mut rev = data.clone();  
    rev.reverse();  
    assert_eq!(rev, vec1![10, 8, 6, 4, 2]);  
    
    // 5. Product  
    let prod: i32 = data.product();  
    assert_eq!(prod, 2*4*6*8*10);  // 3840  
    
    // 6. Prefix sums  
    let prefix = data.prefix_sum_view();  
    assert_eq!(prefix.range_sum(1, 3), 12);   // 2+4+6  
    assert_eq!(prefix.total_sum(), 30);  
    
    // 7. Dot product  
    let other = vec1![1, 2, 3, 4, 5];  
    let dot = data.dot_product(&other);  
    assert_eq!(dot, 2*1 + 4*2 + 6*3 + 8*4 + 10*5);  // 110  
    
    // 8. Sliding window (moving average)  
    let windows: Vec<_> = data.windows(3).collect();  
    assert_eq!(windows[0], &[2, 4, 6]);  
    assert_eq!(windows[1], &[4, 6, 8]);  
    assert_eq!(windows[2], &[6, 8, 10]);  
    
    // 9. Finite difference  
    let diff = data.finite_difference();  
    assert_eq!(diff, vec1![2, 2, 2, 2]);  // Constant difference of arithmetic progression  
    
    // 10. Fourier series  
    use std::f64::consts::PI;  
    let b = vec1![4.0 / PI, 0.0, 4.0 / (3.0 * PI)];  
    let a0 = 0.0;  
    // ... evaluate as needed  
}
📖 Comparison with Standard Vec  
Feature	std::vec::Vec	VecIndexFromOne  
Indexing starts at	0	1  
get(0) returns	Some(&first)	None  
Mathematical formulas	Needs -1 conversion	Direct translation  
no_std support	✅	✅ (with alloc)  
Performance overhead	None	None (zero-cost)  
🧪 Running Tests  
bash  
cargo test  
Test coverage includes initialization, indexing, reverse, product, prefix sums, dot product, windows, finite differences, and more.  

⚙️ no_std Support  
To use this crate in embedded environments without the standard library:  

toml  
[dependencies]  
one-indexed-vec = { version = "0.1.2", default-features = false }  
This enables #![no_std] mode (requires alloc for heap allocation).  

🛡️ Safety Guarantees  
get / get_mut return None for index 0 and out-of-bounds, never panicking  

v[i] / v[i] = x panic with clear error messages showing the 1-based valid range  

All index - 1 conversions use checked_sub, preventing unsigned underflow  

📄 License  
This project is licensed under the MIT License.  

🤝 Contributing  
Contributions are welcome! Please submit issues and pull requests on GitHub.  

Fork the repository  

Create your feature branch (git checkout -b feature/amazing)  

Commit your changes (git commit -m 'Add amazing feature')  

Push to the branch (git push origin feature/amazing)  

Open a Pull Request  

📧 Contact  
Author: may_be_hero  

Email: huanghu578@163.com  

Repository: https://github.com/huanghu578/one-indexed-vec
