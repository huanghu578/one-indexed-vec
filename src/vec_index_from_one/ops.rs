//! Numeric algorithms with 1-based semantics.
//!
//! High-frequency algorithms for numeric elements of [`VecIndexFromOne<T>`]:
//! prefix sum, sum, product, dot product, and non-positive checks.
//!
//! ```rust
//! use one_indexed_vec::{vec1, VecIndexFromOne};
//!
//! let v = vec1![1, 2, 3, 4];
//!
//! let ps = v.prefix_sum();
//! assert_eq!(ps.as_slice(), &[1, 3, 6, 10]); // prefix sum stays 1-based
//!
//! assert_eq!(v.sum(), 10);
//! assert_eq!(v.product(), 24);
//! assert_eq!(v.any_non_positive(), false);
//! ```

use core::ops::{Add, Mul, Sub, Div};

use crate::vec_index_from_one::VecIndexFromOne;
use crate::vec_index_from_one::index1::Index1;

// ============================================================================
// Basic numeric operations
// ============================================================================

impl<T> VecIndexFromOne<T>
where
    T: Copy + Add<Output = T> + Default,
{
    /// Prefix sum (cumulative sum): returns a new container `out` such that
    /// `out[i] = v[1] + v[2] + ... + v[i]`.
    ///
    /// The result is 1-based as well and has the same length as the original.
    /// The initial value is `T::default()` (usually `0` for numeric types).
    ///
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let v = vec1![1, 2, 3];
    /// let ps = v.prefix_sum();
    /// assert_eq!(ps[1], 1);
    /// assert_eq!(ps[2], 3);
    /// assert_eq!(ps[3], 6);
    /// ```
    #[inline]
    pub fn prefix_sum(&self) -> VecIndexFromOne<T> {
        self.prefix_sum_with(T::default())
    }

    /// Prefix sum with an explicit initial accumulator value.
    ///
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let v = vec1![1, 2, 3];
    /// let ps = v.prefix_sum_with(100); // start accumulating from 100
    /// assert_eq!(ps.as_slice(), &[101, 103, 106]);
    /// ```
    #[inline]
    pub fn prefix_sum_with(&self, identity: T) -> VecIndexFromOne<T> {
        let mut out = VecIndexFromOne::with_capacity(self.len());
        let mut acc = identity;
        for &x in self.iter() {
            acc = acc + x;
            out.push(acc);
        }
        out
    }

    /// Sum of all elements.
    #[inline]
    pub fn sum(&self) -> T {
        self.sum_with(T::default())
    }

    /// Sum of all elements with an explicit initial value.
    #[inline]
    pub fn sum_with(&self, identity: T) -> T {
        let mut acc = identity;
        for &x in self.iter() {
            acc = acc + x;
        }
        acc
    }

    /// Computes the cumulative product of all elements.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let v = vec1![2, 3, 4];
    /// let cp = v.cumulative_product();
    /// assert_eq!(cp.as_slice(), &[2, 6, 24]);
    /// ```
    #[inline]
    pub fn cumulative_product(&self) -> VecIndexFromOne<T>
    where
        T: Mul<Output = T>,
    {
        let mut out = VecIndexFromOne::with_capacity(self.len());
        let mut acc = T::default();
        let mut first = true;
        for &x in self.iter() {
            if first {
                acc = x;
                first = false;
            } else {
                acc = acc * x;
            }
            out.push(acc);
        }
        out
    }
}

impl<T> VecIndexFromOne<T>
where
    T: Copy + Mul<Output = T> + From<u8>,
{
    /// Product of all elements.
    ///
    /// Starts from `T::from(1)` (the multiplicative identity), matching
    /// mathematical intuition.
    ///
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let v = vec1![2, 3, 4];
    /// assert_eq!(v.product(), 24);
    /// ```
    #[inline]
    pub fn product(&self) -> T {
        self.product_with(T::from(1))
    }

    /// Product of all elements with an explicit initial value
    /// (e.g. `0`).
    #[inline]
    pub fn product_with(&self, identity: T) -> T {
        let mut acc = identity;
        for &x in self.iter() {
            acc = acc * x;
        }
        acc
    }
}

impl<T> VecIndexFromOne<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    /// Element-wise multiply-accumulate (dot product): `sum(v[i] * w[i])`.
    ///
    /// The two containers must have the same length, otherwise it panics.
    ///
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let a = vec1![1, 2, 3];
    /// let b = vec1![4, 5, 6];
    /// assert_eq!(a.dot_product(&b), 1*4 + 2*5 + 3*6); // 32
    /// ```
    #[inline]
    pub fn dot_product(&self, other: &VecIndexFromOne<T>) -> T {
        assert_eq!(
            self.len(),
            other.len(),
            "dot_product: length mismatch (self len {}, other len {})",
            self.len(),
            other.len()
        );
        let mut acc = T::default();
        for (&x, &y) in self.iter().zip(other.iter()) {
            acc = acc + x * y;
        }
        acc
    }

    /// Computes the dot product up to the 1-based index `i`:
    /// `sum(v[1] * w[1] + v[2] * w[2] + ... + v[i] * w[i])`.
    ///
    /// This is the 1-based equivalent of "take the first i elements".
    ///
    /// # Parameters
    /// - `i`: A 1-based index specifying up to which element to compute.
    ///   - `i = 1` means only the first element (`v[1] * w[1]`)
    ///   - `i = len` means all elements (same as `dot_product`)
    ///
    /// # Panics
    /// Panics if `i == 0`, `i > self.len()`, or `i > other.len()`.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let a = vec1![1, 2, 3, 4];
    /// let b = vec1![5, 6, 7, 8];
    ///
    /// // Compute dot product up to index 3 (1-based):
    /// // v[1]*w[1] + v[2]*w[2] + v[3]*w[3] = 1*5 + 2*6 + 3*7 = 38
    /// assert_eq!(a.dot_product_upto(&b, 3), 38);
    ///
    /// // Compute dot product up to index 1: only v[1]*w[1] = 1*5 = 5
    /// assert_eq!(a.dot_product_upto(&b, 1), 5);
    /// ```
    #[inline]
    pub fn dot_product_upto(&self, other: &VecIndexFromOne<T>, i: usize) -> T {
        assert!(
            i >= 1,
            "dot_product_upto: index i must be >= 1 (got {})",
            i
        );
        assert!(
            i <= self.len(),
            "dot_product_upto: index i ({}) exceeds self length ({})",
            i,
            self.len()
        );
        assert!(
            i <= other.len(),
            "dot_product_upto: index i ({}) exceeds other length ({})",
            i,
            other.len()
        );

        let mut acc = T::default();
        for idx in 1..=i {
            acc = acc + self[idx] * other[idx];
        }
        acc
    }

    /// Computes the dot product up to the 1-based index `i` using the
    /// type-safe [`Index1`] type.
    ///
    /// This is a type-safe alternative to `dot_product_upto`.
    ///
    /// # Example
    /// ```rust    
    /// use one_indexed_vec::{vec1, VecIndexFromOne, Index1};    
    /// let a = vec1![1, 2, 3, 4];
    /// let b = vec1![5, 6, 7, 8];
    /// let result = a.dot_product_upto_index1(&b, Index1::new(3));
    /// assert_eq!(result, 38);
    /// ```
    #[inline]
    pub fn dot_product_upto_index1(&self, other: &VecIndexFromOne<T>, i: Index1) -> T {
        self.dot_product_upto(other, i.get())
    }

    /// Computes the dot product for a range of 1-based indices:
    /// `sum(v[start] * w[start] + ... + v[end] * w[end])`.
    ///
    /// # Parameters
    /// - `start`: 1-based start index (inclusive)
    /// - `end`: 1-based end index (inclusive)
    ///
    /// # Panics
    /// Panics if `start == 0`, `end == 0`, `start > end`,
    /// `end > self.len()`, or `end > other.len()`.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let a = vec1![1, 2, 3, 4];
    /// let b = vec1![5, 6, 7, 8];
    ///
    /// // Compute dot product from index 2 to 4:
    /// // v[2]*w[2] + v[3]*w[3] + v[4]*w[4] = 2*6 + 3*7 + 4*8 = 12 + 21 + 32 = 65
    /// assert_eq!(a.dot_product_range(&b, 2, 4), 65);
    /// ```
    #[inline]
    pub fn dot_product_range(&self, other: &VecIndexFromOne<T>, start: usize, end: usize) -> T {
        assert!(
            start >= 1,
            "dot_product_range: start index must be >= 1 (got {})",
            start
        );
        assert!(
            end >= 1,
            "dot_product_range: end index must be >= 1 (got {})",
            end
        );
        assert!(
            start <= end,
            "dot_product_range: start ({}) must be <= end ({})",
            start,
            end
        );
        assert!(
            end <= self.len(),
            "dot_product_range: end ({}) exceeds self length ({})",
            end,
            self.len()
        );
        assert!(
            end <= other.len(),
            "dot_product_range: end ({}) exceeds other length ({})",
            end,
            other.len()
        );

        let mut acc = T::default();
        for idx in start..=end {
            acc = acc + self[idx] * other[idx];
        }
        acc
    }

    /// Computes the dot product for a range of 1-based indices using
    /// the type-safe [`Index1`] type.
    #[inline]
    pub fn dot_product_range_index1(
        &self,
        other: &VecIndexFromOne<T>,
        start: Index1,
        end: Index1,
    ) -> T {
        self.dot_product_range(other, start.get(), end.get())
    }
}

impl<T> VecIndexFromOne<T>
where
    T: Copy + PartialOrd + Default,
{
    /// Returns whether **any** element is non-positive (`<= T::default()`).
    ///
    /// `Default` is usually `0` for numeric types.
    #[inline]
    pub fn any_non_positive(&self) -> bool {
        let zero = T::default();
        self.iter().any(|&x| x <= zero)
    }

    /// Returns whether **all** elements are non-positive (`<= T::default()`).
    #[inline]
    pub fn all_non_positive(&self) -> bool {
        let zero = T::default();
        self.iter().all(|&x| x <= zero)
    }

    /// Alias for [`any_non_positive`](VecIndexFromOne::any_non_positive),
    /// kept for backward compatibility with earlier naming conventions.
    #[inline]
    pub fn has_non_positive(&self) -> bool {
        self.any_non_positive()
    }
}

// ============================================================================
// Window operations for mathematical and physical formulas
// ============================================================================

impl<T> VecIndexFromOne<T>
where
    T: Copy,
{
    /// Returns a window (slice) from `start` to `end` (1-based, inclusive).
    ///
    /// This is useful for extracting sub-vectors for mathematical operations
    /// like finite differences, convolution, or matrix-vector products.
    ///
    /// # Parameters
    /// - `start`: 1-based start index (inclusive)
    /// - `end`: 1-based end index (inclusive)
    ///
    /// # Panics
    /// Panics if `start < 1`, `end < 1`, `start > end`, or `end > self.len()`.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let v = vec1![1, 2, 3, 4, 5];
    /// let window = v.window(2, 4);
    /// assert_eq!(window.as_slice(), &[2, 3, 4]);
    /// // The window is 1-indexed as well
    /// assert_eq!(window[1], 2);
    /// assert_eq!(window[3], 4);
    /// ```
    #[inline]
    pub fn window(&self, start: usize, end: usize) -> VecIndexFromOne<T> {
        assert!(start >= 1, "window: start must be >= 1 (got {})", start);
        assert!(end >= 1, "window: end must be >= 1 (got {})", end);
        assert!(
            start <= end,
            "window: start ({}) must be <= end ({})",
            start,
            end
        );
        assert!(
            end <= self.len(),
            "window: end ({}) exceeds length ({})",
            end,
            self.len()
        );

        let mut out = VecIndexFromOne::with_capacity(end - start + 1);
        for idx in start..=end {
            out.push(self[idx]);
        }
        out
    }

    /// Returns a mutable window from `start` to `end` (1-based, inclusive).
    ///
    /// This method provides a mutable view into the vector. Modifications
    /// to the returned `WindowMut` will be reflected in the original vector.
    ///
    /// # Parameters
    /// - `start`: 1-based start index (inclusive)
    /// - `end`: 1-based end index (inclusive)
    ///
    /// # Panics
    /// Panics if `start < 1`, `end < 1`, `start > end`, or `end > self.len()`.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let mut v = vec1![1, 2, 3, 4, 5];
    /// {
    ///     let mut window = v.window_mut(2, 4);
    ///     for i in 1..=window.len() {
    ///         window[i] *= 2;
    ///     }
    /// }
    /// assert_eq!(v.as_slice(), &[1, 4, 6, 8, 5]);
    /// ```
    #[inline]
    pub fn window_mut(&mut self, start: usize, end: usize) -> WindowMut<'_, T> {
        assert!(start >= 1, "window_mut: start must be >= 1 (got {})", start);
        assert!(end >= 1, "window_mut: end must be >= 1 (got {})", end);
        assert!(
            start <= end,
            "window_mut: start ({}) must be <= end ({})",
            start,
            end
        );
        assert!(
            end <= self.len(),
            "window_mut: end ({}) exceeds length ({})",
            end,
            self.len()
        );

        WindowMut::new(self, start, end)
    }

    /// Returns a window using type-safe [`Index1`] types.
    #[inline]
    pub fn window_index1(&self, start: Index1, end: Index1) -> VecIndexFromOne<T> {
        self.window(start.get(), end.get())
    }

    /// Returns a mutable window using type-safe [`Index1`] types.
    #[inline]
    pub fn window_mut_index1(&mut self, start: Index1, end: Index1) -> WindowMut<'_, T> {
        self.window_mut(start.get(), end.get())
    }

    /// Returns a view of a window without copying.
    ///
    /// This is more efficient than `window()` when you only need to read.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let v = vec1![1, 2, 3, 4, 5];
    /// let view = v.window_view(2, 4);
    /// assert_eq!(view, &[2, 3, 4]);
    /// ```
    #[inline]
    pub fn window_view(&self, start: usize, end: usize) -> &[T] {
        assert!(start >= 1, "window_view: start must be >= 1 (got {})", start);
        assert!(end >= 1, "window_view: end must be >= 1 (got {})", end);
        assert!(
            start <= end,
            "window_view: start ({}) must be <= end ({})",
            start,
            end
        );
        assert!(
            end <= self.len(),
            "window_view: end ({}) exceeds length ({})",
            end,
            self.len()
        );

        &self.as_slice()[(start - 1)..=end - 1]
    }
}

/// A mutable window view into a `VecIndexFromOne`.
///
/// This struct provides mutable access to a sub-range of the vector.
/// Modifications to elements via this view are reflected in the original vector.
pub struct WindowMut<'a, T> {
    inner: &'a mut VecIndexFromOne<T>,
    start: usize,
    end: usize,
}

impl<'a, T> WindowMut<'a, T> {
    #[inline]
    pub(crate) fn new(inner: &'a mut VecIndexFromOne<T>, start: usize, end: usize) -> Self {
        WindowMut { inner, start, end }
    }

    /// Returns the length of the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.end - self.start + 1
    }

    /// Returns whether the window is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the window as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.inner.as_slice()[(self.start - 1)..self.end]
    }

    /// Returns the window as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.inner.as_mut_slice()[(self.start - 1)..self.end]
    }
}

impl<'a, T> core::ops::Index<usize> for WindowMut<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index >= 1, "WindowMut: index must be >= 1");
        assert!(index <= self.len(), "WindowMut: index out of bounds");
        &self.inner[self.start + index - 1]
    }
}

impl<'a, T> core::ops::IndexMut<usize> for WindowMut<'a, T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index >= 1, "WindowMut: index must be >= 1");
        assert!(index <= self.len(), "WindowMut: index out of bounds");
        &mut self.inner[self.start + index - 1]
    }
}

impl<'a, T> AsRef<[T]> for WindowMut<'a, T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T> AsMut<[T]> for WindowMut<'a, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> VecIndexFromOne<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Default,
{
    /// Computes the finite difference of the vector: `diff[i] = v[i+1] - v[i]`.
    ///
    /// The result has length `len - 1` and is 1-indexed.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let v = vec1![1, 3, 6, 10];
    /// let diff = v.finite_difference();
    /// assert_eq!(diff.as_slice(), &[2, 3, 4]);
    /// assert_eq!(diff[1], 2); // 3 - 1
    /// assert_eq!(diff[2], 3); // 6 - 3
    /// ```
    #[inline]
    pub fn finite_difference(&self) -> VecIndexFromOne<T> {
        if self.len() < 2 {
            return VecIndexFromOne::new();
        }
        let mut out = VecIndexFromOne::with_capacity(self.len() - 1);
        for i in 1..self.len() {
            out.push(self[i + 1] - self[i]);
        }
        out
    }

    /// Computes the second finite difference: `diff2[i] = v[i+2] - 2*v[i+1] + v[i]`.
    ///
    /// The result has length `len - 2` and is 1-indexed.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let v = vec1![1, 4, 9, 16, 25];
    /// let diff2 = v.second_finite_difference();
    /// // For quadratic sequence, second difference is constant
    /// assert_eq!(diff2.as_slice(), &[2, 2, 2]);
    /// ```
    #[inline]
    pub fn second_finite_difference(&self) -> VecIndexFromOne<T> {
        if self.len() < 3 {
            return VecIndexFromOne::new();
        }
        let mut out = VecIndexFromOne::with_capacity(self.len() - 2);
        for i in 1..=self.len() - 2 {
            out.push(self[i + 2] - self[i + 1] - self[i + 1] + self[i]);
        }
        out
    }

    /// Computes the moving average with window size `n`.
    ///
    /// Result length = `len - n + 1`, 1-indexed.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let v = vec1![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let ma = v.moving_average(3);
    /// assert_eq!(ma.as_slice(), &[2.0, 3.0, 4.0]); // (1+2+3)/3, (2+3+4)/3, (3+4+5)/3
    /// ```
    #[inline]
    pub fn moving_average(&self, n: usize) -> VecIndexFromOne<T>
    where
        T: Div<Output = T> + From<u8>,
    {
        assert!(n > 0, "moving_average: window size must be > 0");
        if self.len() < n {
            return VecIndexFromOne::new();
        }

        let factor = T::from(n as u8);
        let mut out = VecIndexFromOne::with_capacity(self.len() - n + 1);
        for start in 1..=self.len() - n + 1 {
            let sum: T = self.window(start, start + n - 1).iter().fold(T::default(), |acc, &x| acc + x);
            out.push(sum / factor);
        }
        out
    }
}

impl<T> VecIndexFromOne<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Div<Output = T> + Default + From<u8>,
{
    /// Computes the central finite difference: `diff[i] = (v[i+1] - v[i-1]) / 2`.
    ///
    /// This is more accurate than forward difference for smooth functions.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// let v = vec1![1.0, 4.0, 9.0, 16.0, 25.0];
    /// let diff = v.central_difference();
    /// // For f(x) = x², central diff approximates 2x
    /// assert_eq!(diff.as_slice(), &[4.0, 6.0, 8.0]);
    /// ```
    #[inline]
    pub fn central_difference(&self) -> VecIndexFromOne<T> {
        if self.len() < 3 {
            return VecIndexFromOne::new();
        }
        let two = T::from(2u8);
        let mut out = VecIndexFromOne::with_capacity(self.len() - 2);
        for i in 2..self.len() {
            out.push((self[i + 1] - self[i - 1]) / two);
        }
        out
    }
}

// ============================================================================
// Sliding window algorithms (filtering, convolution)
// ============================================================================

/// A sliding window iterator over a `VecIndexFromOne`.
///
/// This provides zero-copy iteration over overlapping windows of the vector.
///
/// # Example
/// ```rust
/// use one_indexed_vec::{vec1, VecIndexFromOne};
/// 
/// let v = vec1![1, 2, 3, 4, 5, 6];
/// let mut windows = v.windows(3);
/// 
/// assert_eq!(windows.next().unwrap(), &[1, 2, 3]);
/// assert_eq!(windows.next().unwrap(), &[2, 3, 4]);
/// assert_eq!(windows.next().unwrap(), &[3, 4, 5]);
/// assert_eq!(windows.next().unwrap(), &[4, 5, 6]);
/// assert!(windows.next().is_none());
/// ```
pub struct Windows<'a, T> {
    inner: &'a VecIndexFromOne<T>,
    start: usize,
    end: usize,
}

impl<'a, T> Windows<'a, T> {
    #[inline]
    pub(crate) fn new(inner: &'a VecIndexFromOne<T>, window_size: usize) -> Self {
        Windows {
            inner,
            start: 1,
            end: window_size,
        }
    }
}

impl<'a, T> Iterator for Windows<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.end > self.inner.len() {
            return None;
        }
        
        let slice = &self.inner.as_slice()[(self.start - 1)..self.end];
        self.start += 1;
        self.end += 1;
        Some(slice)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.inner.len().saturating_sub(self.end - 1);
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for Windows<'a, T> {}

/// A mutable sliding window iterator over a `VecIndexFromOne`.
///
/// This allows in-place modification of each window. Changes to the window
/// are reflected in the original vector.
pub struct WindowsMut<'a, T> {
    inner: *mut VecIndexFromOne<T>,
    start: usize,
    end: usize,
    _marker: core::marker::PhantomData<&'a mut T>,
}

impl<'a, T> WindowsMut<'a, T> {
    #[inline]
    pub(crate) fn new(inner: &'a mut VecIndexFromOne<T>, window_size: usize) -> Self {
        WindowsMut {
            inner: inner as *mut VecIndexFromOne<T>,
            start: 1,
            end: window_size,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<'a, T> Iterator for WindowsMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let len = unsafe { (*self.inner).len() };
        if self.end > len {
            return None;
        }
        
        let start_idx = self.start - 1;
        let end_idx = self.end;
        
        let slice = unsafe {
            let ptr = (*self.inner).as_mut_ptr();
            core::slice::from_raw_parts_mut(ptr.add(start_idx), end_idx - start_idx)
        };
        
        self.start += 1;
        self.end += 1;
        Some(unsafe { core::mem::transmute(slice) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = unsafe { (*self.inner).len() };
        let remaining = len.saturating_sub(self.end - 1);
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for WindowsMut<'a, T> {}

impl<T> VecIndexFromOne<T> {
    /// Returns a sliding window iterator over the vector.
    ///
    /// Each window is a slice of length `window_size`. The windows overlap,
    /// sliding by one element each time. This is zero-copy and highly efficient.
    ///
    /// # Panics
    /// Panics if `window_size == 0`.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// 
    /// let v = vec1![1, 2, 3, 4, 5];
    /// let sum_of_windows: Vec<i32> = v.windows(3)
    ///     .map(|w| w.iter().sum())
    ///     .collect();
    /// assert_eq!(sum_of_windows, vec![6, 9, 12]);
    /// ```
    #[inline]
    pub fn windows(&self, window_size: usize) -> Windows<'_, T> {
        assert!(window_size > 0, "windows: window_size must be > 0");
        Windows::new(self, window_size)
    }

    /// Returns a mutable sliding window iterator over the vector.
    ///
    /// This allows in-place modification of each window. Changes to the window
    /// are reflected in the original vector. This is useful for filters,
    /// convolutions, and other signal processing algorithms.
    ///
    /// # Panics
    /// Panics if `window_size == 0`.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// 
    /// let mut v = vec1![1, 2, 3, 4, 5];
    /// 
    /// // Double all elements using windows_mut
    /// for window in v.windows_mut(1) {
    ///     window[0] *= 2;
    /// }
    /// assert_eq!(v.as_slice(), &[2, 4, 6, 8, 10]);
    /// ```
    #[inline]
    pub fn windows_mut(&mut self, window_size: usize) -> WindowsMut<'_, T> {
        assert!(window_size > 0, "windows_mut: window_size must be > 0");
        WindowsMut::new(self, window_size)
    }
}

// ============================================================================
// Prefix sum view (zero-copy cumulative sum)
// ============================================================================

/// A zero-copy view of the prefix sums of a vector.
///
/// This provides O(1) access to the cumulative sum without allocating
/// a new vector, making it ideal for algorithm competitions and data analysis.
///
/// # Example
/// ```rust
/// use one_indexed_vec::{vec1, VecIndexFromOne};
/// 
/// let v = vec1![1, 2, 3, 4, 5];
/// let prefix = v.prefix_sum_view();
/// 
/// // Sum of elements from index 2 to 4: prefix[4] - prefix[1]
/// assert_eq!(prefix.range_sum(2, 4), 9); // 2+3+4 = 9
/// assert_eq!(prefix.total_sum(), 15); // total sum
/// ```
pub struct PrefixSumView<'a, T> {
    inner: &'a VecIndexFromOne<T>,
    cumulative: VecIndexFromOne<T>,
}

impl<'a, T> PrefixSumView<'a, T>
where
    T: Copy + Add<Output = T> + Default,
{
    #[inline]
    pub(crate) fn new(inner: &'a VecIndexFromOne<T>) -> Self {
        let cumulative = inner.prefix_sum();
        PrefixSumView { inner, cumulative }
    }

    /// Returns the prefix sum up to index `i` (1-based, inclusive).
    ///
    /// # Panics
    /// Panics if `i == 0` or `i > len`.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// 
    /// let v = vec1![1, 2, 3, 4];
    /// let prefix = v.prefix_sum_view();
    /// assert_eq!(prefix.get(3), 6); // 1+2+3
    /// ```
    #[inline]
    pub fn get(&self, i: usize) -> T {
        assert!(i >= 1, "PrefixSumView: index must be >= 1");
        assert!(i <= self.inner.len(), "PrefixSumView: index out of bounds");
        self.cumulative[i]
    }

    /// Returns the sum of elements from index `start` to `end` (1-based, inclusive).
    ///
    /// This is O(1) and zero-copy.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// 
    /// let v = vec1![1, 2, 3, 4, 5];
    /// let prefix = v.prefix_sum_view();
    /// assert_eq!(prefix.range_sum(2, 4), 9); // 2+3+4
    /// ```
    #[inline]
    pub fn range_sum(&self, start: usize, end: usize) -> T
    where
        T: Sub<Output = T>,
    {
        assert!(start >= 1, "PrefixSumView: start must be >= 1");
        assert!(end >= start, "PrefixSumView: end must be >= start");
        assert!(end <= self.inner.len(), "PrefixSumView: end out of bounds");
        
        if start == 1 {
            self.get(end)
        } else {
            self.get(end) - self.get(start - 1)
        }
    }

    /// Returns the total sum of all elements.
    #[inline]
    pub fn total_sum(&self) -> T {
        if self.inner.is_empty() {
            T::default()
        } else {
            self.cumulative[self.inner.len()]
        }
    }
}

impl<T> VecIndexFromOne<T>
where
    T: Copy + Add<Output = T> + Default,
{
    /// Creates a zero-copy prefix sum view.
    ///
    /// This is useful when you need to answer many range sum queries
    /// without allocating a new vector.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// 
    /// let v = vec1![1, 2, 3, 4, 5];
    /// let prefix = v.prefix_sum_view();
    /// 
    /// // O(1) range sum queries
    /// assert_eq!(prefix.range_sum(2, 4), 9);
    /// assert_eq!(prefix.range_sum(1, 5), 15);
    /// assert_eq!(prefix.range_sum(3, 3), 3);
    /// ```
    #[inline]
    pub fn prefix_sum_view(&self) -> PrefixSumView<'_, T> {
        PrefixSumView::new(self)
    }

    /// Returns the prefix sum as a new vector (allocates).
    ///
    /// This is a convenience alias for `prefix_sum()`.
    #[inline]
    pub fn cumsum_alloc(&self) -> VecIndexFromOne<T> {
        self.prefix_sum()
    }
}

// ============================================================================
// Capacity management
// ============================================================================

impl<T> VecIndexFromOne<T> {
    /// Shrinks the capacity to fit the current length if the capacity is
    /// significantly larger than the length.
    ///
    /// The `factor` parameter controls how aggressive the shrink is:
    /// - `factor = 2`: shrinks if capacity > 2 * length (common threshold)
    /// - `factor = 4`: shrinks if capacity > 4 * length (less aggressive)
    /// - `factor = 1`: always shrinks (same as `shrink_to_fit`)
    ///
    /// This is useful in production environments where you want to avoid
    /// repeated allocations while still freeing memory when appropriate.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// 
    /// let mut v = VecIndexFromOne::with_capacity(1000);
    /// v.extend(vec1![1, 2, 3, 4, 5].iter().copied());
    /// 
    /// // Capacity is 1000, length is 5
    /// assert!(v.capacity() >= 1000);
    /// 
    /// // Shrink if capacity > 4 * length (1000 > 20)
    /// v.shrink_to_fit_if(4);
    /// assert!(v.capacity() < 20);
    /// ```
    #[inline]
    pub fn shrink_to_fit_if(&mut self, factor: usize) {
        if factor == 0 {
            self.shrink_to_fit();
            return;
        }
        
        let len = self.len();
        let cap = self.capacity();
        
        if cap > len * factor {
            self.shrink_to_fit();
        }
    }

    /// Shrinks the capacity to fit the current length if the capacity exceeds
    /// the length by at least `threshold` elements.
    ///
    /// This is useful when you know the maximum size you'll need but want
    /// to free memory when usage drops significantly.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// 
    /// let mut v = VecIndexFromOne::with_capacity(1000);
    /// v.extend(vec1![1, 2, 3, 4, 5].iter().copied());
    /// 
    /// // Shrink if capacity - length > 100 (true: 1000 - 5 = 995 > 100)
    /// v.shrink_to_fit_if_excess(100);
    /// assert!(v.capacity() < 100);
    /// ```
    #[inline]
    pub fn shrink_to_fit_if_excess(&mut self, threshold: usize) {
        let len = self.len();
        let cap = self.capacity();
        
        if cap > len && cap - len > threshold {
            self.shrink_to_fit();
        }
    }

    /// Ensures the vector has at least `min_capacity` capacity.
    ///
    /// This is like `reserve` but only grows, never shrinks.
    /// Returns the new capacity.
    ///
    #[inline]
    pub fn ensure_capacity(&mut self, min_capacity: usize) -> usize {
        if self.capacity() < min_capacity {
            self.reserve(min_capacity - self.capacity());
        }
        self.capacity()
    }
}

// ============================================================================
// Additional utility functions
// ============================================================================

impl<T> VecIndexFromOne<T>
where
    T: Copy + PartialEq,
{
    /// Returns the index of the first occurrence of `value` (1-based).
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// 
    /// let v = vec1![10, 20, 30, 20, 40];
    /// assert_eq!(v.find_first(&20), Some(2));
    /// assert_eq!(v.find_first(&50), None);
    /// ```
    #[inline]
    pub fn find_first(&self, value: &T) -> Option<usize> {
        self.iter().position(|x| x == value).map(|pos| pos + 1)
    }

    /// Returns the index of the last occurrence of `value` (1-based).
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// 
    /// let v = vec1![10, 20, 30, 20, 40];
    /// assert_eq!(v.find_last(&20), Some(4));
    /// ```
    #[inline]
    pub fn find_last(&self, value: &T) -> Option<usize> {
        self.iter().rposition(|x| x == value).map(|pos| pos + 1)
    }
}

impl<T> VecIndexFromOne<T>
where
    T: Copy + Add<Output = T> + Default,
{
    /// Computes the mean (average) of all elements.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// 
    /// let v: VecIndexFromOne<f64> = vec1![1.0, 2.0, 3.0, 4.0, 5.0];
    /// assert!((v.mean() - 3.0).abs() < 1e-10);
    /// ```
    #[inline]
    pub fn mean(&self) -> T
    where
        T: Div<Output = T> + From<u8>,
    {
        if self.is_empty() {
            T::default()
        } else {
            self.sum() / T::from(self.len() as u8)
        }
    }

    /// Computes the variance of all elements.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// 
    /// let v: VecIndexFromOne<f64> = vec1![1.0, 2.0, 3.0, 4.0, 5.0];
    /// assert!((v.variance() - 2.0).abs() < 1e-10);
    /// ```
    #[inline]
    pub fn variance(&self) -> T
    where
        T: Div<Output = T> + From<u8> + Sub<Output = T> + Mul<Output = T>,
    {
        if self.len() < 2 {
            T::default()
        } else {
            let mean = self.mean();
            let sum_sq = self.iter().fold(T::default(), |acc, &x| {
                let diff = x - mean;
                acc + diff * diff
            });
            sum_sq / T::from(self.len() as u8)
        }
    }

    /// Computes the standard deviation of all elements.
    ///
    /// # Example
    /// ```rust
    /// use one_indexed_vec::{vec1, VecIndexFromOne};
    /// 
    /// let v: VecIndexFromOne<f64> = vec1![1.0, 2.0, 3.0, 4.0, 5.0];
    /// assert!((v.std_dev() - 1.41421356).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn std_dev(&self) -> T
    where
        T: Div<Output = T> + From<u8> + Sub<Output = T> + Mul<Output = T>,
        f64: From<T>,
        T: From<f64>,
    {
        if self.len() < 2 {
            T::default()
        } else {
            let var = self.variance();
            let var_f64: f64 = var.into();
            let std_f64 = var_f64.sqrt();
            T::from(std_f64)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec1;

    // ---- prefix sum tests ----

    #[test]
    fn prefix_sum_basic() {
        let v = vec1![1, 2, 3, 4];
        let ps = v.prefix_sum();
        assert_eq!(ps.as_slice(), &[1, 3, 6, 10]);
        assert_eq!(ps[1], 1);
        assert_eq!(ps[4], 10);
    }

    #[test]
    fn prefix_sum_empty() {
        let v: VecIndexFromOne<i32> = VecIndexFromOne::new();
        assert!(v.prefix_sum().is_empty());
    }

    #[test]
    fn prefix_sum_with_identity() {
        let v = vec1![1, 2, 3];
        assert_eq!(v.prefix_sum_with(100).as_slice(), &[101, 103, 106]);
    }

    #[test]
    fn prefix_sum_f64() {
        let v = vec1![1.5f64, 2.5];
        let ps = v.prefix_sum();
        assert!((ps[1] - 1.5).abs() < 1e-10);
        assert!((ps[2] - 4.0).abs() < 1e-10);
    }

    // ---- cumulative product tests ----

    #[test]
    fn cumulative_product_basic() {
        let v = vec1![2, 3, 4];
        let cp = v.cumulative_product();
        assert_eq!(cp.as_slice(), &[2, 6, 24]);
    }

    #[test]
    fn cumulative_product_empty() {
        let v: VecIndexFromOne<i32> = VecIndexFromOne::new();
        assert!(v.cumulative_product().is_empty());
    }

    #[test]
    fn cumulative_product_single() {
        let v = vec1![5];
        let cp = v.cumulative_product();
        assert_eq!(cp.as_slice(), &[5]);
    }

    // ---- sum/product tests ----

    #[test]
    fn sum_works() {
        let v = vec1![1, 2, 3, 4];
        assert_eq!(v.sum(), 10);
        assert_eq!(v.sum_with(100), 110);
    }

    #[test]
    fn product_works() {
        let v = vec1![2, 3, 4];
        assert_eq!(v.product(), 24);
    }

    #[test]
    fn product_with_identity() {
        let v = vec1![2, 3, 4];
        assert_eq!(v.product_with(1), 24);
        assert_eq!(v.product_with(0), 0);
    }

    // ---- dot product tests ----

    #[test]
    fn dot_product_works() {
        let a = vec1![1, 2, 3];
        let b = vec1![4, 5, 6];
        assert_eq!(a.dot_product(&b), 32);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn dot_product_length_mismatch_panics() {
        let a = vec1![1, 2];
        let b = vec1![1, 2, 3];
        let _ = a.dot_product(&b);
    }

    #[test]
    fn dot_product_upto_basic() {
        let a = vec1![1, 2, 3, 4];
        let b = vec1![5, 6, 7, 8];
        assert_eq!(a.dot_product_upto(&b, 3), 38);
        assert_eq!(a.dot_product_upto(&b, 1), 5);
    }

    #[test]
    fn dot_product_upto_equals_full_dot_product() {
        let a = vec1![1, 2, 3, 4, 5];
        let b = vec1![6, 7, 8, 9, 10];

        assert_eq!(
            a.dot_product_upto(&b, a.len()),
            a.dot_product(&b)
        );
    }

    #[test]
    fn dot_product_upto_with_negative_numbers() {
        let a = vec1![-1, 2, -3, 4];
        let b = vec1![5, -6, 7, -8];

        assert_eq!(a.dot_product_upto(&b, 2), -17);
        assert_eq!(a.dot_product_upto(&b, 4), -70);
    }

    #[test]
    #[should_panic(expected = "index i must be >= 1")]
    fn dot_product_upto_zero_panics() {
        let a = vec1![1, 2, 3];
        let b = vec1![4, 5, 6];
        let _ = a.dot_product_upto(&b, 0);
    }

    #[test]
    fn dot_product_range_basic() {
        let a = vec1![1, 2, 3, 4, 5];
        let b = vec1![6, 7, 8, 9, 10];
        assert_eq!(a.dot_product_range(&b, 2, 4), 74);
    }

    #[test]
    fn dot_product_range_full_range() {
        let a = vec1![1, 2, 3];
        let b = vec1![4, 5, 6];

        assert_eq!(
            a.dot_product_range(&b, 1, 3),
            a.dot_product(&b)
        );
    }

    #[test]
    fn dot_product_range_with_negative_numbers() {
        let a = vec1![-1, 2, -3, 4];
        let b = vec1![5, -6, 7, -8];

        assert_eq!(a.dot_product_range(&b, 2, 3), -33);
    }

    // ---- window tests ----

    #[test]
    fn window_basic() {
        let v = vec1![1, 2, 3, 4, 5];
        let w = v.window(2, 4);
        assert_eq!(w.as_slice(), &[2, 3, 4]);
        assert_eq!(w[1], 2);
        assert_eq!(w[3], 4);
    }

    #[test]
    fn window_single_element() {
        let v = vec1![1, 2, 3];
        let w = v.window(2, 2);
        assert_eq!(w.as_slice(), &[2]);
        assert_eq!(w[1], 2);
    }

    #[test]
    fn window_full_range() {
        let v = vec1![1, 2, 3, 4];
        let w = v.window(1, 4);
        assert_eq!(w.as_slice(), v.as_slice());
    }

    #[test]
    #[should_panic(expected = "window: start must be >= 1")]
    fn window_start_zero_panics() {
        let v = vec1![1, 2, 3];
        let _ = v.window(0, 2);
    }

    #[test]
    #[should_panic(expected = "window: end must be >= 1")]
    fn window_end_zero_panics() {
        let v = vec1![1, 2, 3];
        let _ = v.window(1, 0);
    }

    #[test]
    #[should_panic(expected = "window: start (3) must be <= end (2)")]
    fn window_start_gt_end_panics() {
        let v = vec1![1, 2, 3];
        let _ = v.window(3, 2);
    }

    #[test]
    #[should_panic(expected = "window: end (4) exceeds length (3)")]
    fn window_exceeds_length_panics() {
        let v = vec1![1, 2, 3];
        let _ = v.window(1, 4);
    }

    #[test]
    fn window_mut_modifies_original() {
        let mut v = vec1![1, 2, 3, 4, 5];
        {
            let mut window = v.window_mut(2, 4);
            for i in 1..=window.len() {
                window[i] *= 2;
            }
        }
        assert_eq!(v.as_slice(), &[1, 4, 6, 8, 5]);
    }

    #[test]
    fn window_view_basic() {
        let v = vec1![1, 2, 3, 4, 5];
        let view = v.window_view(2, 4);
        assert_eq!(view, &[2, 3, 4]);
    }

    #[test]
    fn window_index1_type_safe() {
        use crate::vec_index_from_one::index1::Index1;
        let v = vec1![1, 2, 3, 4, 5];
        let w = v.window_index1(Index1::new(2), Index1::new(4));
        assert_eq!(w.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn window_mut_index1_type_safe() {
        use crate::vec_index_from_one::index1::Index1;
        let mut v = vec1![1, 2, 3, 4, 5];
        {
            let mut window = v.window_mut_index1(Index1::new(2), Index1::new(4));
            for i in 1..=window.len() {
                window[i] *= 2;
            }
        }
        assert_eq!(v.as_slice(), &[1, 4, 6, 8, 5]);
    }

    // ---- windows iterator tests ----

    #[test]
    fn windows_iterator_basic() {
        let v = vec1![1, 2, 3, 4, 5];
        let mut windows = v.windows(3);
        
        assert_eq!(windows.next().unwrap(), &[1, 2, 3]);
        assert_eq!(windows.next().unwrap(), &[2, 3, 4]);
        assert_eq!(windows.next().unwrap(), &[3, 4, 5]);
        assert!(windows.next().is_none());
    }

    #[test]
    fn windows_iterator_size_hint() {
        let v = vec1![1, 2, 3, 4, 5, 6];
        let windows = v.windows(3);
        let (min, max) = windows.size_hint();
        assert_eq!(min, 4);
        assert_eq!(max, Some(4));
    }

    #[test]
    fn windows_iterator_exact_size() {
        let v = vec1![1, 2, 3, 4];
        let windows = v.windows(2);
        assert_eq!(windows.len(), 3);
    }

    #[test]
    fn windows_mut_basic() {
        let mut v = vec1![1, 2, 3, 4, 5];
        
        // Double all elements using windows_mut with window size 1
        for window in v.windows_mut(1) {
            window[0] *= 2;
        }
        assert_eq!(v.as_slice(), &[2, 4, 6, 8, 10]);
    }

    // ---- finite difference tests ----

    #[test]
    fn finite_difference_basic() {
        let v = vec1![1, 3, 6, 10];
        let diff = v.finite_difference();
        assert_eq!(diff.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn finite_difference_empty() {
        let v: VecIndexFromOne<i32> = VecIndexFromOne::new();
        assert!(v.finite_difference().is_empty());
        
        let v = vec1![5];
        assert!(v.finite_difference().is_empty());
    }

    #[test]
    fn second_finite_difference_basic() {
        let v = vec1![1, 4, 9, 16, 25];
        let diff2 = v.second_finite_difference();
        assert_eq!(diff2.as_slice(), &[2, 2, 2]);
    }

    #[test]
    fn second_finite_difference_empty() {
        let v = vec1![1, 2];
        assert!(v.second_finite_difference().is_empty());
    }

    #[test]
    fn central_difference_basic() {
        let v: VecIndexFromOne<f64> = vec1![1.0, 4.0, 9.0, 16.0, 25.0];
        let diff = v.central_difference();
        assert_eq!(diff.as_slice(), &[4.0, 6.0, 8.0]);
    }

    #[test]
    fn central_difference_empty_for_small_vector() {
        let v = vec1![1.0, 2.0];
        assert!(v.central_difference().is_empty());
    }

    // ---- moving average tests ----

    #[test]
    fn moving_average_basic() {
        let v: VecIndexFromOne<f64> = vec1![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = v.moving_average(3);
        assert_eq!(ma.as_slice(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn moving_average_window_size_one() {
        let v: VecIndexFromOne<f64> = vec1![1.0, 2.0, 3.0];
        let ma = v.moving_average(1);
        assert_eq!(ma.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn moving_average_empty_for_small_vector() {
        let v: VecIndexFromOne<f64> = vec1![1.0, 2.0];
        let ma = v.moving_average(3);
        assert!(ma.is_empty());
    }

    // ---- prefix_sum_view tests ----

    #[test]
    fn prefix_sum_view_basic() {
        let v = vec1![1, 2, 3, 4, 5];
        let prefix = v.prefix_sum_view();
        
        assert_eq!(prefix.get(1), 1);
        assert_eq!(prefix.get(3), 6);
        assert_eq!(prefix.get(5), 15);
    }

    #[test]
    fn prefix_sum_view_range_sum() {
        let v = vec1![1, 2, 3, 4, 5];
        let prefix = v.prefix_sum_view();
        
        assert_eq!(prefix.range_sum(2, 4), 9);
        assert_eq!(prefix.range_sum(1, 5), 15);
        assert_eq!(prefix.range_sum(3, 3), 3);
    }

    #[test]
    fn prefix_sum_view_total_sum() {
        let v = vec1![1, 2, 3, 4, 5];
        let prefix = v.prefix_sum_view();
        assert_eq!(prefix.total_sum(), 15);
        
        let empty: VecIndexFromOne<i32> = VecIndexFromOne::new();
        let prefix = empty.prefix_sum_view();
        assert_eq!(prefix.total_sum(), 0);
    }

    #[test]
    #[should_panic(expected = "PrefixSumView: index must be >= 1")]
    fn prefix_sum_view_index_zero_panics() {
        let v = vec1![1, 2, 3];
        let prefix = v.prefix_sum_view();
        let _ = prefix.get(0);
    }

    // ---- shrink tests ----

    #[test]
    fn shrink_to_fit_if_basic() {
        let mut v = VecIndexFromOne::with_capacity(1000);
        v.extend(vec1![1, 2, 3, 4, 5].iter().copied());
        
        let original_cap = v.capacity();
        assert!(original_cap >= 1000);
        
        v.shrink_to_fit_if(4);
        assert!(v.capacity() < 20);
    }

    #[test]
    fn shrink_to_fit_if_does_not_shrink_when_factor_high() {
        let mut v = VecIndexFromOne::with_capacity(100);
        v.extend(vec1![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].iter().copied());
        
        v.shrink_to_fit_if(20);
        assert!(v.capacity() >= 100);
    }

    #[test]
    fn shrink_to_fit_if_excess_basic() {
        let mut v = VecIndexFromOne::with_capacity(1000);
        v.extend(vec1![1, 2, 3].iter().copied());
        
        v.shrink_to_fit_if_excess(10);
        assert!(v.capacity() < 100);
    }

    #[test]
    fn shrink_to_fit_if_excess_no_shrink() {
        let mut v = VecIndexFromOne::with_capacity(20);
        v.extend(vec1![1, 2, 3, 4, 5].iter().copied());
        
        v.shrink_to_fit_if_excess(20);
        assert!(v.capacity() >= 20);
    }

#[test]
fn ensure_capacity_grows_when_needed() {
    let mut v = VecIndexFromOne::new();
    
    // 先添加一些元素
    for i in 1..=3 {
        v.push(i);
    }
    
    let old_cap = v.capacity();
    let new_cap = v.ensure_capacity(old_cap + 50);
    
    // 确保容量增长了
    assert!(new_cap > old_cap);
    assert!(v.capacity() > old_cap);
}

    #[test]
    fn ensure_capacity_no_shrink() {
        let mut v = VecIndexFromOne::with_capacity(100);
        v.extend(vec1![1, 2, 3].iter().copied());
        
        let cap = v.ensure_capacity(10);
        assert!(cap >= 100);
    }

    // ---- find methods tests ----

    #[test]
    fn find_first_basic() {
        let v = vec1![10, 20, 30, 20, 40];
        assert_eq!(v.find_first(&20), Some(2));
        assert_eq!(v.find_first(&50), None);
    }

    #[test]
    fn find_last_basic() {
        let v = vec1![10, 20, 30, 20, 40];
        assert_eq!(v.find_last(&20), Some(4));
        assert_eq!(v.find_last(&50), None);
    }

    // ---- mean/variance/std_dev tests ----

    #[test]
    fn mean_basic() {
        let v: VecIndexFromOne<f64> = vec1![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((v.mean() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn mean_empty() {
        let v: VecIndexFromOne<f64> = VecIndexFromOne::new();
        assert!((v.mean() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn variance_basic() {
        let v: VecIndexFromOne<f64> = vec1![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((v.variance() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn variance_single_element() {
        let v: VecIndexFromOne<f64> = vec1![5.0];
        assert!((v.variance() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn std_dev_basic() {
        let v: VecIndexFromOne<f64> = vec1![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((v.std_dev() - 1.41421356).abs() < 1e-6);
    }

    // ---- non-positive checks ----

    #[test]
    fn non_positive_checks() {
        let neg = vec1![-1, -2, -3];
        assert!(neg.all_non_positive());
        assert!(neg.any_non_positive());

        let mixed = vec1![-1, 5];
        assert!(!mixed.all_non_positive());
        assert!(mixed.any_non_positive());
        assert!(mixed.has_non_positive());

        let pos = vec1![1, 2, 3];
        assert!(!pos.any_non_positive());
    }

    // ---- edge cases ----

    #[test]
    fn empty_container_operations() {
        let v: VecIndexFromOne<i32> = VecIndexFromOne::new();
        assert_eq!(v.sum(), 0);
        assert_eq!(v.product(), 1);
        assert!(!v.any_non_positive());
        assert!(v.all_non_positive());
        assert!(v.prefix_sum().is_empty());
    }

    #[test]
    fn single_element_operations() {
        let v = vec1![5];
        assert_eq!(v.sum(), 5);
        assert_eq!(v.product(), 5);
        assert!(!v.any_non_positive());
        assert!(!v.all_non_positive());

        let v = vec1![-5];
        assert!(v.any_non_positive());
        assert!(v.all_non_positive());
    }

    #[test]
    fn empty_container_prefix_sum_view() {
        let v: VecIndexFromOne<i32> = VecIndexFromOne::new();
        let prefix = v.prefix_sum_view();
        assert_eq!(prefix.total_sum(), 0);
    }

    #[test]
    fn dot_product_upto_index1_type_safe() {
        use crate::vec_index_from_one::index1::Index1;

        let a = vec1![1, 2, 3, 4];
        let b = vec1![5, 6, 7, 8];

        assert_eq!(
            a.dot_product_upto_index1(&b, Index1::new(3)),
            a.dot_product_upto(&b, 3)
        );
    }

    #[test]
    fn dot_product_range_index1_type_safe() {
        use crate::vec_index_from_one::index1::Index1;

        let a = vec1![1, 2, 3, 4];
        let b = vec1![5, 6, 7, 8];

        assert_eq!(
            a.dot_product_range_index1(&b, Index1::new(2), Index1::new(4)),
            a.dot_product_range(&b, 2, 4)
        );
    }
}