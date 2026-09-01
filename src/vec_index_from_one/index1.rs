//! Type-safe indexing with [`Index1`].
//!
//! A bare `usize` cannot tell a "0-based subscript" apart from a "1-based
//! position", which makes mixing them easy to get wrong. [`Index1`] wraps a
//! 1-based position in its own type to rule out such confusion:
//!
//! ```rust
//! use one_indexed_vec::{Index1, VecIndexFromOne};
//!
//! let v = VecIndexFromOne::from(vec![10, 20, 30]);
//! let pos = Index1::new(2);                // the 2nd position
//! assert_eq!(v[pos], 20);                  // direct subscript
//! assert_eq!(pos.to_zero_based(), Some(1)); // underlying 0-based index
//! ```

use core::fmt;

/// A 1-based position index.
///
/// Internally stores a position value starting at `1`:
/// `Index1::new(1)` refers to the first element.
/// This is a `Copy` plain value type and can be freely copied and compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Index1(usize);

impl Index1 {
    /// Constructs from a 1-based position.
    ///
    /// `Index1::new(1)` is the first element. Passing `0` does not fail here,
    /// but the position is treated as invalid when used to index a container
    /// (`to_zero_based` returns `None`).
    #[inline]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Constructs from a 1-based position (equivalent to [`Index1::new`],
    /// with a more explicit name).
    #[inline]
    pub const fn from_one_based(index: usize) -> Self {
        Self(index)
    }

    /// Returns the 1-based position value.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }

    /// Converts to the underlying 0-based subscript.
    ///
    /// Position `1` maps to subscript `0`; the invalid position `0` returns
    /// `None` (never underflows).
    #[inline]
    pub const fn to_zero_based(self) -> Option<usize> {
        self.0.checked_sub(1)
    }

    /// Moves the position toward the tail, saturating at `usize::MAX`.
    #[inline]
    pub const fn saturating_add(self, rhs: usize) -> Self {
        Self(self.0.saturating_add(rhs))
    }

    /// Moves the position toward the head, with a lower bound of `1`
    /// (it never drops to `0`).
    #[inline]
    pub const fn saturating_sub(self, rhs: usize) -> Self {
        let v = self.0.saturating_sub(rhs);
        Self(if v < 1 { 1 } else { v })
    }
}

impl fmt::Display for Index1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Index1> for usize {
    #[inline]
    fn from(i: Index1) -> Self {
        i.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_get_roundtrip() {
        let i = Index1::new(5);
        assert_eq!(i.get(), 5);
        assert_eq!(Index1::from_one_based(5), i);
    }

    #[test]
    fn to_zero_based_conversion() {
        assert_eq!(Index1::new(1).to_zero_based(), Some(0));
        assert_eq!(Index1::new(3).to_zero_based(), Some(2));
        assert_eq!(Index1::new(0).to_zero_based(), None); // invalid position
    }

    #[test]
    fn saturating_arithmetic() {
        assert_eq!(Index1::new(3).saturating_add(2).get(), 5);
        assert_eq!(Index1::new(3).saturating_sub(2).get(), 1);
        assert_eq!(Index1::new(1).saturating_sub(10).get(), 1); // lower bound 1
    }

    #[test]
    fn copy_and_compare() {
        let a = Index1::new(2);
        let b = a; // Copy
        assert_eq!(a, b);
        assert!(Index1::new(1) < Index1::new(2));
        assert_eq!(Index1::default(), Index1::new(0));
    }

    #[test]
    fn display_and_from_usize() {
        assert_eq!(Index1::new(42).to_string(), "42");
        let n: usize = Index1::new(7).into();
        assert_eq!(n, 7);
    }
}
