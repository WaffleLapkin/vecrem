//! Cursor-like helper which allows removing elements from vector without
//! moving the tail every time.
//!
//! ## `Vec::remove` comparison
//!
//! If you'll use common `Vec::remove` to remove only some elements starting
//! from the head, you'll have bad times because it will copy a lot of elements:
//! ```text
//! vec's memory: [0, 1, 2, 3, 4]
//!
//! > vec.remove(0);
//! vec's memory: [-, 1, 2, 3, 4]
//! vec's memory: [1, 2, 3, 4, -] // copy of 4 elements (the whole tail)
//!
//! > vec.remove(1);
//! vec's memory: [1, -, 3, 4, -]
//! vec's memory: [1, 3, 4, -, -] // copy of 2 elements
//!
//! > vec.remove(2);
//! vec's memory: [1, 3, -, -, -]
//! ```
//!
//! Whereas [`Removing`] uses `swap`s:
//!
//! ```text
//! vec's memory: [0, 1, 2, 3, 4]
//! rem's ptr:     ^
//!
//! > let rem = vec.removing();
//! > rem.next().unwrap().remove();
//! vec's memory: [-, 1, 2, 3, 4]
//! rem's ptr:        ^
//!
//! > rem.next().unwrap().skip();
//! vec's memory: [1, -, 2, 3, 4] // one copy of 1
//! rem's ptr:           ^
//!
//! > rem.next().unwrap().remove();
//! vec's memory: [1, -, -, 3, 4]
//! rem's ptr:              ^
//!
//! > rem.next().unwrap().skip();
//! vec's memory: [1, 3, -, -, 4] // one copy of 3
//! rem's ptr:                 ^
//!
//! > rem.next().unwrap().remove();
//! vec's memory: [1, 3, -, -, -]
//! ```
//!
//! ## no_std support
//!
//! This crate supports `#![no_std]` but requires `alloc` (we are working with
//! vec after all)
#![deny(missing_docs)]
#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::ptr;

/// A cursor-like structure made for cheap removing of elements from vector.
///
/// Essentially overhead of the process is one `ptr::copy` for each [`skip`]ed
/// element + one `ptr::copy` for the not yielded tail.
///
/// For the comparison with manual [`Vec::remove`] see [self#]
///
/// Also, an analog to this may be [`VecDeque`] (if you need to only pop
/// elements from either of ends) or (unstable as of writing this)
/// [`Vec::drain_filter`] (if you need to remove all matching elements).
///
/// However note 2 differences between [`Vec::drain_filter`] and [`Removing`]:
/// - [`Removing`] is not iterator (this makes it more universal but less
///   convenient to use)
/// - [`Removing`] does **not** remove elements from the vec when you drop it
///
/// The main method of this struct is [`next`] which returns [`Entry`] which can
/// be used to mutate the vec.
///
/// ## Examples
///
/// ```
/// use vecrem::VecExt;
///
/// let mut vec: Vec<_> = (0..17).collect();
/// let mut out = Vec::new();
/// {
///     let mut rem = vec.removing();
///
///     while let Some(entry) = rem.next() {
///         let value = *entry.value();
///         if value >= 10 {
///             break;
///         }
///
///         if value % 2 == 0 {
///             out.push(entry.remove());
///         } else {
///             entry.skip();
///         }
///     }
/// }
///
/// // All `skip`ed and not yielded elements are in the original vec
/// assert_eq!(vec, [1, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16]);
///
/// assert_eq!(out, [0, 2, 4, 6, 8])
/// ```
///
/// ## Leak Notes
///
/// In the same way as [`Vec::drain_filter`] for efishient work, [`Removing`]
/// needs to remporary break vec leaving it in incunsisntent state.
/// The state is made normal in `Drop`, however in rust running destructors is
/// not guaranteed (see [`mem::forget`]). As such, [`Removing`] sets vec's len
/// to `0` (and restores it in `Drop`), this means that if [`Removing`] gets
/// leaked or forgotten - the elements of the vectore are gone too.
pub struct Removing<'a, T> {
    // Type invariants:
    // - vec.capacity() >= len
    // - vec[..slot] is initialized (note: dye to safety guarantees vec.len is
    //   set to 0, so this is not 'real' indexing)
    // - vec[curr..len] is initialized
    // - vec[..len] is the same allocation
    vec: &'a mut Vec<T>,
    slot: usize,
    curr: usize,
    len: usize,
}

impl<'a, T> Removing<'a, T> {
    /// Creates new [`Removing`] instance from given vec.
    ///
    /// See also: [`VecExt::removing`]
    #[inline]
    pub fn new(vec: &'a mut Vec<T>) -> Self {
        let len = vec.len();

        // ## Safety
        //
        // - 0 <= vec.len() for any vec
        // - vec.len()..0 is always none elements => all needed elements are initialize
        unsafe {
            // This is needed to safely break vec invariants & not expose safety bags when
            vec.set_len(0);
        }

        Self {
            vec,
            slot: 0,
            curr: 0,
            len,
        }
    }

    /// Returns [`Entry`] for the next element.
    #[inline]
    pub fn next(&mut self) -> Option<Entry<'_, 'a, T>> {
        if self.is_empty() {
            None
        } else {
            Some(Entry { rem: self })
        }
    }

    /// Return `true` if all elements of the underling vector were either
    /// [`skip`]ed or [`remove`]d (i.e.: when [`next`] will return `Some(_)`)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.curr >= self.len
    }

    // private
    /// ## Safety
    ///
    /// offset **must** be `<= vec.capacity()`
    unsafe fn ptr_mut(&mut self, offset: usize) -> *mut T {
        // Safety must be uphold be the caller
        self.vec.as_mut_ptr().add(offset)
    }

    fn slot_mut(&mut self) -> *mut T {
        // ## Safety
        //
        // `slot` always points to the content of the vector
        unsafe { self.ptr_mut(self.slot) }
    }

    fn curr_mut(&mut self) -> *mut T {
        // ## Safety
        //
        // `curr` always points to the content of the vector
        unsafe { self.ptr_mut(self.curr) }
    }
}

impl<T> Drop for Removing<'_, T> {
    fn drop(&mut self) {
        unsafe {
            if !self.is_empty() {
                // Copy the tail
                // [A, B, -, -, -, C, D] => [A, B, C, D, -, C*, D*]
                // * logically this memory is uninitialized i.e. it's a move
                ptr::copy(self.curr_mut(), self.slot_mut(), self.len - self.curr);
                self.slot += self.len - self.curr;
            }

            // Slot points to the first uninitialized element
            self.vec.set_len(self.slot)
        }
    }
}

/// Vec entry.
///
/// Entry allows you to [`remove`], [`skip`], [read] or [mutate] the element
/// behind it.
///
/// Note: if an entry is not used (neither of [`remove`], [`skip`] is called on it)
/// the `Removing::next` method will yield the same entry again.
///
/// The only way to get this struct is to call [`Removing::next`] method.
#[must_use = "You should either remove an entry, or skip it"]
pub struct Entry<'a, 'rem, T> {
    rem: &'a mut Removing<'rem, T>,
}

impl<T> Entry<'_, '_, T> {
    /// Remove element behind this entry.
    #[inline]
    pub fn remove(mut self) -> T {
        unsafe {
            let curr = self.curr_mut();
            self.rem.curr += 1;

            ptr::read(curr)
        }
    }

    /// Skip element behind this entry, leaving it in the vec.
    #[inline]
    pub fn skip(mut self) {
        unsafe {
            ptr::copy(self.curr_mut(), self.slot_mut(), 1);
            self.rem.slot += 1;
            self.rem.curr += 1;
        }
    }

    /// Get access to the element behind this entry.
    #[inline]
    pub fn value(&self) -> &T {
        unsafe { &*self.curr_ptr() }
    }

    /// Get unique access to the element behind this entry.
    #[inline]
    pub fn value_mut(&mut self) -> &mut T {
        unsafe { &mut *self.curr_mut() }
    }

    /// Get unique access to the element after the element behind this entry.
    ///
    /// Returns `None` if this entry corresponds to the last item in the vec.
    #[inline]
    pub fn peek_next(&mut self) -> Option<&mut T> {
        let next = self.rem.curr + 1;

        if next >= self.rem.len {
            return None;
        }

        unsafe { Some(&mut *self.rem.ptr_mut(next)) }
    }

    // private

    fn curr_mut(&mut self) -> *mut T {
        self.rem.curr_mut()
    }

    fn curr_ptr(&self) -> *const T {
        // ## Safety
        //
        // `curr` always points to the content of the vector
        unsafe { self.rem.vec.as_ptr().add(self.rem.curr) }
    }

    fn slot_mut(&mut self) -> *mut T {
        self.rem.slot_mut()
    }
}

/// Extension for [`Vec`](alloc::Vec) which adds [`removing`] method
pub trait VecExt<T> {
    /// Creates new [`Removing`] instance from given vec.
    fn removing(&mut self) -> Removing<T>;
}

impl<T> VecExt<T> for Vec<T> {
    #[inline]
    fn removing(&mut self) -> Removing<T> {
        Removing::new(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::VecExt;

    use core::mem;
    use std::fmt::Debug;

    /// Returns non-copy type that can help miri detect safety bugs
    fn f(i: i32) -> impl Debug + Eq {
        i.to_string()
    }

    #[test]
    fn clear() {
        let mut vec: Vec<_> = (0..10).map(f).collect();
        let mut out = Vec::with_capacity(10);
        let mut rem = vec.removing();

        while let Some(entry) = rem.next() {
            out.push(entry.remove());
        }

        assert_eq!(out, (0..10).map(f).collect::<Vec<_>>())
    }

    #[test]
    fn skip() {
        let mut vec: Vec<_> = (0..10).map(f).collect();
        let mut rem = vec.removing();

        while let Some(entry) = rem.next() {
            entry.skip();
        }
    }

    #[test]
    fn drop_entry() {
        let mut vec = vec![f(0)];
        let mut rem = vec.removing();
        let mut timeout = 0..100;

        while let Some(_) = rem.next().zip(timeout.next()) {}

        assert!(timeout.is_empty());
    }

    #[test]
    fn forget_entry() {
        let mut vec = vec![f(0)];
        let mut rem = vec.removing();
        let mut timeout = 0..100;

        while let Some((entry, _)) = rem.next().zip(timeout.next()) {
            mem::forget(entry)
        }

        assert!(timeout.is_empty());
    }

    #[test]
    fn forget() {
        let mut vec: Vec<_> = (0..10).map(f).collect();
        let mut rem = vec.removing();

        if let Some(entry) = rem.next() {
            entry.remove();
        }
        if let Some(entry) = rem.next() {
            entry.remove();
        }
        if let Some(entry) = rem.next() {
            entry.remove();
        }

        mem::forget(rem);

        assert_eq!(vec, []);
    }

    #[test]
    fn drop() {
        let mut vec: Vec<_> = (0..10).map(f).collect();
        mem::drop(vec.removing());
        assert_eq!(vec, (0..10).map(f).collect::<Vec<_>>());
    }

    #[test]
    fn even() {
        let mut vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut out = Vec::with_capacity(10);
        let mut rem = vec.removing();

        while let Some(entry) = rem.next() {
            if *entry.value() % 2 == 0 {
                out.push(entry.remove());
            } else {
                entry.skip();
            }
        }

        mem::drop(rem);

        assert_eq!(out, [0, 2, 4, 6, 8]);
        assert_eq!(vec, [1, 3, 5, 7, 9]);
    }

    #[test]
    fn break_() {
        let mut vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut rem = vec.removing();

        while let Some(entry) = rem.next() {
            if *entry.value() >= 5 {
                break;
            }

            entry.remove();
        }

        mem::drop(rem);

        assert_eq!(vec, [5, 6, 7, 8, 9]);
    }

    #[test]
    fn test() {
        let mut vec = vec![0, 1, 2];

        {
            let mut rem = vec.removing();
            rem.next().unwrap().skip();
            assert_eq!(rem.next().unwrap().remove(), 1);
            rem.next().unwrap().skip();
        }

        assert_eq!(vec, [0, 2]);
    }
}
