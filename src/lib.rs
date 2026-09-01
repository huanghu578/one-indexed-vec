//! # one-indexed-vec
//!
//! A one-based (1-indexed) wrapper around `Vec<T>`.
//!
//! The standard library's `Vec<T>` indexes from `0`; the [`VecIndexFromOne<T>`]
//! type provided by this crate instead places the first element at index `1`,
//! matching the 1-based conventions of Fortran, MATLAB, spreadsheets, and many
//! mathematical formulas.
//!
//! ## Quick start
//!
//! ```rust
//! use one_indexed_vec::{VecIndexFromOne, vec1};
//!
//! let mut v = VecIndexFromOne::from(vec![10, 20, 30]);
//! v.push(40);                    // append
//! assert_eq!(v.len(), 4);
//! assert_eq!(v[1], 10);          // indexing starts at 1
//! assert_eq!(v.get(0), None);    // 0 is an invalid index
//! assert_eq!(v.first(), Some(&10));
//!
//! let v2 = vec1![1, 2, 3];
//! let dpu = v.dot_product_upto(&v2, 3);   //10*1+20*2+30*3=140
//! assert_eq!(dpu, 140);
//! ```
//!
//! ## Type-safe indexing
//!
//! When you want to guarantee at compile time that a 0-based index cannot be
//! passed to a 1-based container, use the [`Index1`] strong type:
//!
//! ```rust
//! use one_indexed_vec::{VecIndexFromOne, Index1, vec1};
//!
//! let v = vec1![10, 20, 30];
//! let first = Index1::new(1);   // explicit 1-based position
//! assert_eq!(v[first], 10);
//! ```
//!
//! ## Numeric algorithms
//!
//! High-frequency numeric algorithms with 1-based semantics are provided for
//! numeric elements: prefix sum, element-wise product, non-positive checks, etc.
//!
//! ### Prefix sum and range queries
//!
//! The `prefix_sum_view` provides O(1) range sum queries without allocation:
//!
//! ```rust
//! use one_indexed_vec::{vec1, VecIndexFromOne};
//!
//! let v = vec1![1, 2, 3, 4, 5];
//! let prefix = v.prefix_sum_view();
//!
//! // Sum of elements from index 2 to 4: 2 + 3 + 4 = 9
//! assert_eq!(prefix.range_sum(2, 4), 9);
//! assert_eq!(prefix.total_sum(), 15);
//! ```
//!
//! ### Sliding windows
//!
//! Zero-copy sliding window iteration for filtering and convolution:
//!
//! ```rust
//! use one_indexed_vec::{vec1, VecIndexFromOne};
//!
//! let v = vec1![1, 2, 3, 4, 5, 6];
//! let mut windows = v.windows(3);
//!
//! assert_eq!(windows.next().unwrap(), &[1, 2, 3]);
//! assert_eq!(windows.next().unwrap(), &[2, 3, 4]);
//! assert_eq!(windows.next().unwrap(), &[3, 4, 5]);
//! assert_eq!(windows.next().unwrap(), &[4, 5, 6]);
//! ```
//!
//! ### Mutable windows for in-place modification
//!
//! ```rust
//! use one_indexed_vec::{vec1, VecIndexFromOne};
//!
//! let mut v = vec1![1, 2, 3, 4, 5];
//!
//! // Double all elements using windows_mut with window size 1
//! for window in v.windows_mut(1) {
//!     window[0] *= 2;
//! }
//! assert_eq!(v.as_slice(), &[2, 4, 6, 8, 10]);
//!
//! // Or use window_mut for a specific range
//! {
//!     let mut window = v.window_mut(2, 4);
//!     for i in 1..=window.len() {
//!         window[i] += 1;
//!     }
//! }
//! assert_eq!(v.as_slice(), &[2, 5, 7, 9, 10]);
//! ```
//!
//! ### Finite differences
//!
//! Compute derivatives of discretized functions:
//!
//! ```rust
//! use one_indexed_vec::{vec1, VecIndexFromOne};
//!
//! let v = vec1![1, 4, 9, 16, 25];
//! let diff = v.finite_difference();
//! assert_eq!(diff.as_slice(), &[3, 5, 7, 9]);  // f(x) = x², diff ≈ 2x+1
//!
//! let diff2 = v.second_finite_difference();
//! assert_eq!(diff2.as_slice(), &[2, 2, 2]);    // second difference is constant
//! ```
//!
//! ### Moving average (smoothing)
//!
//! ```rust
//! use one_indexed_vec::{vec1, VecIndexFromOne};
//!
//! let v = vec1![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
//! let smoothed = v.moving_average(3);
//! assert_eq!(smoothed.as_slice(), &[2.0, 3.0, 4.0, 5.0, 6.0]);
//! ```
//!
//! ### Fourier series evaluation
//!
//! The 1-indexed vector type is particularly useful for Fourier series where
//! coefficients are naturally indexed from 1:
//!
//! ```rust
//! use one_indexed_vec::{VecIndexFromOne, vec1};
//! use std::f64::consts::PI;
//!
//! /// Evaluates the Fourier series at point x
//! /// 
//! /// a0: constant term (a₀/2)
//! /// a: cosine coefficient vector (a₁, a₂, a₃, ...)
//! /// b: sine coefficient vector (b₁, b₂, b₃, ...)
//! fn fourier_eval(x: f64, a0: f64, a: &VecIndexFromOne<f64>, b: &VecIndexFromOne<f64>) -> f64 {
//!     let max_len = a.len().max(b.len());
//!     let mut result = a0;
//!     
//!     for n in 1..=max_len {
//!         let nx = n as f64 * x;
//!         let a_n = if n <= a.len() { a[n] } else { 0.0 };
//!         let b_n = if n <= b.len() { b[n] } else { 0.0 };
//!         result += a_n * nx.cos() + b_n * nx.sin();
//!     }
//!     
//!     result
//! }
//!
//! // Square wave Fourier series coefficients (first 10 terms)
//! // Square wave: f(x) = 1 for x in (0, π), f(x) = -1 for x in (-π, 0)
//! let a0 = 0.0;                        // DC component
//! let a = vec1![];                      // No cosine terms for square wave
//! let b = vec1![
//!     4.0 / PI,                        // b₁
//!     0.0,                             // b₂
//!     4.0 / (3.0 * PI),                // b₃
//!     0.0,                             // b₄
//!     4.0 / (5.0 * PI),                // b₅
//!     0.0,                             // b₆
//!     4.0 / (7.0 * PI),                // b₇
//!     0.0,                             // b₈
//!     4.0 / (9.0 * PI),                // b₉
//!     0.0,                             // b₁₀
//! ];
//!
//! // Evaluate square wave at x = π/2
//! let value = fourier_eval(PI / 2.0, a0, &a, &b);
//! println!("f(π/2) = {}", value);      // Should be close to 1.0
//! ```
//!
//! ### Capacity management
//!
//! Optimize memory usage with conditional shrinking:
//!
//! ```rust
//! use one_indexed_vec::{vec1, VecIndexFromOne};
//!
//! let mut v = VecIndexFromOne::with_capacity(1000);
//! v.extend(vec1![1, 2, 3, 4, 5].iter().copied());
//!
//! // Shrink if capacity > 4 * length
//! v.shrink_to_fit_if(4);
//! assert!(v.capacity() < 20);
//!
//! // Or shrink if excess capacity > threshold
//! let mut v2 = VecIndexFromOne::with_capacity(1000);
//! v2.extend(vec1![1, 2, 3].iter().copied());
//! v2.shrink_to_fit_if_excess(10);
//! assert!(v2.capacity() < 100);
//! ```
//!
//! ## no_std
//!
//! `std` is enabled by default; disabling the default features
//! (`default-features = false`) switches to `no_std` (only `alloc` is needed).
//!
//! ## Safety
//!
//! - [`get`](VecIndexFromOne::get) / [`get_mut`](VecIndexFromOne::get_mut)
//!   return `None` for index `0` and out-of-bounds indices, never panicking;
//! - the subscript syntax `v[i]` / `v[i] = x` (`Index` / `IndexMut`) panics on
//!   index `0` and out-of-bounds indices, with the 1-based valid range in the
//!   error message;
//! - every internal `index - 1` conversion goes through `checked_sub`, so
//!   unsigned underflow can never happen.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod vec_index_from_one;

pub use vec_index_from_one::index1::Index1;
pub use vec_index_from_one::ops;
pub use vec_index_from_one::VecIndexFromOne;



// Re-export window and iterator types for external use
pub use vec_index_from_one::ops::{WindowMut, Windows, WindowsMut, PrefixSumView};