//! `VecIndexFromOne<T>` — a one-based wrapper around `Vec<T>`.
//!
//! Unlike the standard library's `Vec<T>` (0-based), the first element of this
//! type lives at index `1`:
//!
//! ```rust
//! use one_indexed_vec::{vec1, VecIndexFromOne};
//!
//! let mut v = vec1!["a", "b"];
//! // or
//! // let mut v = VecIndexFromOne::from(vec!["a", "b"]);
//!
//! assert_eq!(v[1], "a"); // indexing starts at 1
//! assert_eq!(v[2], "b");
//! assert_eq!(v.len(), 2);
//! ```
//!
//! Index `0` and out-of-bounds indices are invalid: `get`/`get_mut` return
//! `None`, while `Index`/`IndexMut` (the `v[i]` syntax) panic.
//!
//! Submodules:
//! - [`index1`]: the type-safe index [`Index1`];
//! - [`ops`]: numeric algorithms (prefix sum, dot product, etc.).

pub mod index1;
pub mod ops;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::ops::{Index, IndexMut};

use self::index1::Index1;

/// A one-based wrapper around `Vec<T>`.
///
/// Internally holds a `Vec<T>`, but every index exposed to the caller starts
/// at `1`: the first element is `vec[1]` and the last is `vec[len]`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(transparent)
)]
pub struct VecIndexFromOne<T> {
    inner: Vec<T>,
}

impl<T> VecIndexFromOne<T> {
    /// Constructs an empty container.
    #[inline]
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    /// Constructs an empty container with the given capacity (in elements).
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { inner: Vec::with_capacity(capacity) }
    }

    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the container is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Appends an element to the tail.
    #[inline]
    pub fn push(&mut self, value: T) {
        self.inner.push(value);
    }

    /// Removes and returns the last element.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    /// Clears all elements.
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Inserts `value` at the 1-based position `index` (`1 <= index <= len + 1`).
    ///
    /// - `insert(1, x)` is equivalent to inserting at the very front;
    /// - `insert(len + 1, x)` is equivalent to `push(x)`.
    ///
    /// Panics when out of bounds (`index == 0` or `index > len + 1`).
    pub fn insert(&mut self, index: usize, value: T) {
        let pos = index
            .checked_sub(1)
            .unwrap_or_else(|| panic!("insertion index (is {index}) should be >= 1"));
        self.inner.insert(pos, value);
    }

    /// Removes and returns the element at the 1-based position `index`
    /// (`1 <= index <= len`).
    ///
    /// Panics when out of bounds (`index == 0` or `index > len`).
    pub fn remove(&mut self, index: usize) -> T {
        let pos = index
            .checked_sub(1)
            .unwrap_or_else(|| panic!("removal index (is {index}) should be >= 1"));
        self.inner.remove(pos)
    }

    /// Returns an immutable reference to the element at the 1-based position
    /// `index`.
    ///
    /// Returns `None` when `index == 0` or `index > len`.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        index.checked_sub(1).and_then(|pos| self.inner.get(pos))
    }

    /// Returns a mutable reference to the element at the 1-based position
    /// `index`.
    ///
    /// Returns `None` when `index == 0` or `index > len`.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        index.checked_sub(1).and_then(|pos| self.inner.get_mut(pos))
    }

    /// Returns an immutable reference to the element at the [`Index1`]
    /// position (type-safe indexing).
    #[inline]
    pub fn get_index1(&self, index: Index1) -> Option<&T> {
        self.get(index.get())
    }

    /// Returns a mutable reference to the element at the [`Index1`] position
    /// (type-safe indexing).
    #[inline]
    pub fn get_index1_mut(&mut self, index: Index1) -> Option<&mut T> {
        self.get_mut(index.get())
    }

    /// Returns the first element (equivalent to `get(1)`).
    #[inline]
    pub fn first(&self) -> Option<&T> {
        self.inner.first()
    }

    /// Returns a mutable reference to the first element.
    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.inner.first_mut()
    }

    /// Returns the last element (equivalent to `get(len)`).
    #[inline]
    pub fn last(&self) -> Option<&T> {
        self.inner.last()
    }

    /// Returns a mutable reference to the last element.
    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.inner.last_mut()
    }

    /// Iterates over the elements in order (immutable).
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.inner.iter()
    }

    /// Iterates over the elements in order (mutable).
    #[inline]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.inner.iter_mut()
    }

    /// Returns an immutable iterator carrying 1-based indices.
    ///
    /// The first element is yielded with index `1`, the `len`-th element with
    /// index `len`.
    #[inline]
    pub fn iter_indexed(&self) -> impl Iterator<Item = (usize, &T)> {
        self.inner.iter().enumerate().map(|(i, v)| (i + 1, v))
    }

    /// Returns the range of 1-based indices, `1..=len` (indices only).
    #[inline]
    pub fn indices(&self) -> core::ops::RangeInclusive<usize> {
        1..=self.len()
    }

    /// Returns a new `VecIndexFromOne` with the elements in reverse order.
    ///
    /// # Example
    /// ```
    /// use one_indexed_vec::VecIndexFromOne;
    ///
    /// let v = VecIndexFromOne::from(vec![1, 2, 3]);
    /// let rev = v.reversed();
    /// assert_eq!(rev[1], 3);
    /// assert_eq!(rev[2], 2);
    /// assert_eq!(rev[3], 1);
    /// ```
    #[inline]
    pub fn reversed(&self) -> Self
    where
        T: Clone,
    {
        let mut inner = self.inner.clone();
        inner.reverse();
        Self { inner }
    }

    /// Reverses the order of elements in place.
    ///
    /// # Example
    /// ```
    /// use one_indexed_vec::VecIndexFromOne;
    ///
    /// let mut v = VecIndexFromOne::from(vec![1, 2, 3]);
    /// v.reverse();
    /// assert_eq!(v[1], 3);
    /// assert_eq!(v[2], 2);
    /// assert_eq!(v[3], 1);
    /// ```
    #[inline]
    pub fn reverse(&mut self) {
        self.inner.reverse();
    }

    /// Returns a reference to the underlying `Vec<T>`.
    #[inline]
    pub fn as_vec(&self) -> &Vec<T> {
        &self.inner
    }

    /// Views all elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.inner
    }
    /// Returns the number of elements the vector can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Reserves capacity for at least `additional` more elements.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Shrinks the capacity of the vector as much as possible.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }

    /// Returns a mutable slice of the entire vector.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }

    /// Returns a mutable pointer to the vector's buffer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr()
    }
}

impl<T> From<Vec<T>> for VecIndexFromOne<T> {
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        Self { inner: vec }
    }
}

impl<T> From<VecIndexFromOne<T>> for Vec<T> {
    #[inline]
    fn from(v: VecIndexFromOne<T>) -> Self {
        v.inner
    }
}

impl<T> FromIterator<T> for VecIndexFromOne<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self { inner: iter.into_iter().collect() }
    }
}

impl<T> Extend<T> for VecIndexFromOne<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.inner.extend(iter);
    }
}

impl<T: Clone> VecIndexFromOne<T> {
    /// Replaces the element at position `index` with `value` and returns the
    /// old value.
    ///
    /// Panics when `index == 0` or `index > len`.
    pub fn replace(&mut self, index: usize, value: T) -> T {
        let pos = index
            .checked_sub(1)
            .unwrap_or_else(|| panic!("replacement index (is {index}) should be >= 1"));
        core::mem::replace(&mut self.inner[pos], value)
    }
}

/// Supports the `v[i]` subscript syntax (1-based).
///
/// Index `0` and out-of-bounds indices both panic, with an error message
/// following the standard library's style.
impl<T> Index<usize> for VecIndexFromOne<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).unwrap_or_else(|| {
            panic!(
                "index out of bounds: the len is {} but the index is {index} (one-based: valid range is 1..={})",
                self.len(),
                self.len()
            )
        })
    }
}

/// Supports the `v[i] = x` subscript assignment syntax (1-based).
impl<T> IndexMut<usize> for VecIndexFromOne<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let len = self.len();
        self.get_mut(index).unwrap_or_else(|| {
            panic!(
                "index out of bounds: the len is {len} but the index is {index} (one-based: valid range is 1..={len})"
            )
        })
    }
}

/// Supports the `v[index1]` subscript syntax (type-safe, 1-based).
///
/// Internally reuses the `Index<usize>` implementation; invalid positions such
/// as `Index1::new(0)` panic as well.
impl<T> Index<Index1> for VecIndexFromOne<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: Index1) -> &Self::Output {
        &self[index.get()]
    }
}

/// Supports the `v[index1] = x` subscript assignment syntax (type-safe index).
impl<T> IndexMut<Index1> for VecIndexFromOne<T> {
    #[inline]
    fn index_mut(&mut self, index: Index1) -> &mut Self::Output {
        &mut self[index.get()]
    }
}

/// Creates a `VecIndexFromOne` containing the given elements.
///
/// The first element will be at index `1`.
///
/// # Example
/// ```
/// use one_indexed_vec::{vec1, VecIndexFromOne};
///
/// let v = vec1![1, 2, 3];
/// assert_eq!(v[1], 1);
/// assert_eq!(v[2], 2);
/// assert_eq!(v[3], 3);
///
/// let empty: VecIndexFromOne<i32> = vec1![];
/// assert!(empty.is_empty());
/// ```
#[macro_export]
macro_rules! vec1 {
    // 空向量
    () => {
        $crate::VecIndexFromOne::new()  // 保持与原类型名一致
    };
    // 重复语法：vec1![0.0; 6]
    ($elem:expr; $n:expr) => {{
        let n = $n;
        let mut v = $crate::VecIndexFromOne::with_capacity(n);  // 使用正确的类型名
        for _ in 0..n {
            v.push($elem.clone());
        }
        v
    }};
    // 列举元素：vec1![1, 2, 3]
    ($($x:expr),+ $(,)?) => {{
        let mut v = $crate::VecIndexFromOne::new();
        $(
            v.push($x);
        )*
        v
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- construction ----

    #[test]
    fn new_is_empty() {
        let v: VecIndexFromOne<i32> = VecIndexFromOne::new();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn from_vec_rebases_first_element_to_one() {
        let v = VecIndexFromOne::from(vec![10, 20, 30]);
        assert_eq!(v.len(), 3);
        assert_eq!(v.get(1), Some(&10)); // original 0th -> 1st
        assert_eq!(v.get(2), Some(&20));
        assert_eq!(v.get(3), Some(&30));
    }

    #[test]
    fn from_impl_and_into_inner() {
        let v: VecIndexFromOne<i32> = Vec::from(vec![1, 2]).into();
        assert_eq!(v.len(), 2);
        let back: Vec<i32> = v.into();
        assert_eq!(back, vec![1, 2]);
    }

    #[test]
    fn collect_from_iterator() {
        let v: VecIndexFromOne<i32> = (1..=5).collect();
        assert_eq!(v.len(), 5);
        assert_eq!(v[1], 1);
        assert_eq!(v[5], 5);
    }

    // ---- access ----

    #[test]
    fn get_with_valid_index() {
        let v = VecIndexFromOne::from(vec!["a", "b", "c"]);
        assert_eq!(v.get(1), Some(&"a"));
        assert_eq!(v.get(3), Some(&"c"));
    }

    #[test]
    fn get_zero_returns_none() {
        let v = VecIndexFromOne::from(vec![1, 2, 3]);
        assert_eq!(v.get(0), None); // 0 is an invalid index
    }

    #[test]
    fn get_out_of_bounds_returns_none() {
        let v = VecIndexFromOne::from(vec![1, 2, 3]);
        assert_eq!(v.get(4), None);
        assert_eq!(v.get(usize::MAX), None);
    }

    #[test]
    fn get_mut_allows_mutation() {
        let mut v = VecIndexFromOne::from(vec![1, 2, 3]);
        if let Some(x) = v.get_mut(2) {
            *x = 99;
        }
        assert_eq!(v[2], 99);
        assert_eq!(v.get_mut(0), None);
    }

    #[test]
    fn first_and_last() {
        let v = VecIndexFromOne::from(vec![1, 2, 3]);
        assert_eq!(v.first(), Some(&1));
        assert_eq!(v.last(), Some(&3));
        let empty: VecIndexFromOne<i32> = VecIndexFromOne::new();
        assert_eq!(empty.first(), None);
        assert_eq!(empty.last(), None);
    }

    // ---- Index syntax ----

    #[test]
    fn index_syntax_is_one_based() {
        let v = VecIndexFromOne::from(vec![10, 20, 30]);
        assert_eq!(v[1], 10);
        assert_eq!(v[2], 20);
        assert_eq!(v[3], 30);
    }

    #[test]
    fn index_mut_syntax() {
        let mut v = VecIndexFromOne::from(vec![1, 2, 3]);
        v[2] = 42;
        assert_eq!(v[2], 42);
    }

    #[test]
    #[should_panic(expected = "one-based: valid range is 1..=")]
    fn index_zero_panics() {
        let v = VecIndexFromOne::from(vec![1, 2, 3]);
        let _ = v[0];
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn index_out_of_bounds_panics() {
        let v = VecIndexFromOne::from(vec![1, 2, 3]);
        let _ = v[4];
    }

    #[test]
    #[should_panic(expected = "one-based")]
    fn index_mut_zero_panics() {
        let mut v = VecIndexFromOne::from(vec![1]);
        v[0] = 9;
    }

    // ---- mutation ----

    #[test]
    fn push_increases_len_and_appends() {
        let mut v = VecIndexFromOne::new();
        v.push(1);
        v.push(2);
        assert_eq!(v.len(), 2);
        assert_eq!(v[1], 1);
        assert_eq!(v[2], 2);
    }

    #[test]
    fn pop_returns_last_and_shrinks() {
        let mut v = VecIndexFromOne::from(vec![1, 2, 3]);
        assert_eq!(v.pop(), Some(3));
        assert_eq!(v.len(), 2);
        assert_eq!(v.pop(), Some(2));
        assert_eq!(v.pop(), Some(1));
        assert_eq!(v.pop(), None);
        assert!(v.is_empty());
    }

    #[test]
    fn insert_at_front_remains_one_based() {
        let mut v = VecIndexFromOne::from(vec![2, 3]);
        v.insert(1, 1); // insert at the very front
        assert_eq!(v.len(), 3);
        assert_eq!(v[1], 1);
        assert_eq!(v[2], 2);
        assert_eq!(v[3], 3);
    }

    #[test]
    fn insert_at_len_plus_one_equals_push() {
        let mut v = VecIndexFromOne::from(vec![1, 2]);
        v.insert(3, 3);
        assert_eq!(v[3], 3);
        assert_eq!(v.len(), 3);
    }

    #[test]
    #[should_panic(expected = "should be >= 1")]
    fn insert_at_zero_panics() {
        let mut v = VecIndexFromOne::from(vec![1]);
        v.insert(0, 99);
    }

    #[test]
    fn remove_at_one_based_position() {
        let mut v = VecIndexFromOne::from(vec![1, 2, 3]);
        assert_eq!(v.remove(2), 2);
        assert_eq!(v.len(), 2);
        assert_eq!(v[1], 1);
        assert_eq!(v[2], 3); // later elements shift forward
    }

    #[test]
    fn replace_swaps_value() {
        let mut v = VecIndexFromOne::from(vec![1, 2, 3]);
        let old = v.replace(2, 99);
        assert_eq!(old, 2);
        assert_eq!(v[2], 99);
    }

    #[test]
    fn clear_empties() {
        let mut v = VecIndexFromOne::from(vec![1, 2, 3]);
        v.clear();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    // ---- iteration ----

    #[test]
    fn iter_yields_elements_in_order() {
        let v = VecIndexFromOne::from(vec![1, 2, 3]);
        let collected: Vec<_> = v.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn iter_indexed_starts_at_one() {
        let v = VecIndexFromOne::from(vec![10, 20, 30]);
        let pairs: Vec<_> = v.iter_indexed().collect();
        assert_eq!(pairs, vec![(1, &10), (2, &20), (3, &30)]);
    }

    #[test]
    fn indices_are_one_to_len_inclusive() {
        let v = VecIndexFromOne::from(vec![10, 20, 30]);
        let idx: Vec<_> = v.indices().collect();
        assert_eq!(idx, vec![1, 2, 3]);
    }

    #[test]
    fn iter_mut_allows_mutation() {
        let mut v = VecIndexFromOne::from(vec![1, 2, 3]);
        for x in v.iter_mut() {
            *x *= 10;
        }
        assert_eq!(v.as_slice(), &[10, 20, 30]);
    }

    // ---- extension ----

    #[test]
    fn extend_appends_batch() {
        let mut v = VecIndexFromOne::from(vec![1, 2]);
        v.extend([3, 4, 5]);
        assert_eq!(v.len(), 5);
        assert_eq!(v[5], 5);
    }

    // ---- edge cases ----

    #[test]
    fn large_index_does_not_underflow() {
        // ensure get(usize::MAX) never panics from subtraction underflow
        let v = VecIndexFromOne::from(vec![1]);
        assert_eq!(v.get(usize::MAX), None);
        let mut v = VecIndexFromOne::from(vec![1]);
        assert_eq!(v.get_mut(usize::MAX), None);
    }

    // ---- type-safe indexing with Index1 ----

    #[test]
    fn index1_syntax_reads_elements() {
        let v = VecIndexFromOne::from(vec![10, 20, 30]);
        assert_eq!(v[Index1::new(1)], 10);
        assert_eq!(v[Index1::new(2)], 20);
        assert_eq!(v[Index1::new(3)], 30);
    }

    #[test]
    fn index1_syntax_writes_elements() {
        let mut v = VecIndexFromOne::from(vec![1, 2, 3]);
        v[Index1::new(2)] = 99;
        assert_eq!(v[Index1::new(2)], 99);
    }

    #[test]
    fn get_index1_methods() {
        let mut v = VecIndexFromOne::from(vec![1, 2, 3]);
        assert_eq!(v.get_index1(Index1::new(1)), Some(&1));
        assert_eq!(v.get_index1(Index1::new(0)), None); // invalid position
        if let Some(x) = v.get_index1_mut(Index1::new(3)) {
            *x = 30;
        }
        assert_eq!(v[Index1::new(3)], 30);
    }

    #[test]
    #[should_panic(expected = "one-based")]
    fn index1_zero_panics() {
        let v = VecIndexFromOne::from(vec![1]);
        let _ = v[Index1::new(0)];
    }

    // ---- reverse ----

    #[test]
    fn reversed_creates_new_reversed_copy() {
        let v = VecIndexFromOne::from(vec![1, 2, 3, 4]);
        let rev = v.reversed();
        assert_eq!(rev.len(), 4);
        assert_eq!(rev[1], 4);
        assert_eq!(rev[2], 3);
        assert_eq!(rev[3], 2);
        assert_eq!(rev[4], 1);
        // original unchanged
        assert_eq!(v[1], 1);
        assert_eq!(v[4], 4);
    }

    #[test]
    fn reverse_in_place() {
        let mut v = VecIndexFromOne::from(vec!['a', 'b', 'c', 'd']);
        v.reverse();
        assert_eq!(v[1], 'd');
        assert_eq!(v[2], 'c');
        assert_eq!(v[3], 'b');
        assert_eq!(v[4], 'a');
    }

    #[test]
    fn reverse_empty() {
        let mut v: VecIndexFromOne<i32> = VecIndexFromOne::new();
        v.reverse();
        assert!(v.is_empty());
        let rev = v.reversed();
        assert!(rev.is_empty());
    }

    #[test]
    fn reverse_single_element() {
        let mut v = VecIndexFromOne::from(vec![42]);
        v.reverse();
        assert_eq!(v[1], 42);
        let rev = v.reversed();
        assert_eq!(rev[1], 42);
    }

    // ---- vec1! macro ----

    #[test]
    fn vec1_macro_empty() {
        let v: VecIndexFromOne<i32> = vec1![];
        assert!(v.is_empty());
    }

    #[test]
    fn vec1_macro_single() {
        let v = vec1![42];
        assert_eq!(v.len(), 1);
        assert_eq!(v[1], 42);
    }

    #[test]
    fn vec1_macro_multiple() {
        let v = vec1![1, 2, 3, 4, 5];
        assert_eq!(v.len(), 5);
        assert_eq!(v[1], 1);
        assert_eq!(v[3], 3);
        assert_eq!(v[5], 5);
    }

    #[test]
    fn vec1_macro_trailing_comma() {
        let v = vec1!['a', 'b', 'c',];
        assert_eq!(v.len(), 3);
        assert_eq!(v[1], 'a');
        assert_eq!(v[3], 'c');
    }

    #[test]
    fn vec1_macro_with_expressions() {
        let v = vec1![1 + 1, 2 * 3, 10 - 5];
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 6);
        assert_eq!(v[3], 5);
    }
}