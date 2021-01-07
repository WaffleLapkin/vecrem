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
//! > rem.next().unwrap();
//! vec's memory: [1, -, 2, 3, 4] // one copy of 1
//! rem's ptr:           ^
//!
//! > rem.next().unwrap().remove();
//! vec's memory: [1, -, -, 3, 4]
//! rem's ptr:              ^
//!
//! > rem.next().unwrap();
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
use core::{mem::ManuallyDrop, ptr};

// FIXME(waffle): add support for backwards iteration

/// A cursor-like structure made for cheap iterative removing of elements from
/// vector.
///
/// Essentially overhead of the process is one `ptr::copy` for each ignored
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
/// [`VecDeque`]: alloc::collections::VecDeque
/// [`next`]: Removing::next
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
///         }
///     }
/// }
///
/// // All ignored and not yielded elements are in the original vec
/// assert_eq!(vec, [1, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16]);
///
/// assert_eq!(out, [0, 2, 4, 6, 8])
/// ```
///
/// ## forget behavior
///
/// In the same way as [`Vec::drain_filter`], for efficient work, [`Removing`]
/// needs to temporarily break vec's invariants leaving it in an inconsistent
/// state. The state is made normal in `Drop`.
///
/// However in rust running destructors is not guaranteed (see [`mem::forget`],
/// [`ManuallyDrop`]). As such, on construction [`Removing`] sets vec's len to
/// `0` (and restores it in `Drop`), this means that if [`Removing`] gets leaked
/// or forgotten - the elements of the vector are forgotten too.
///
/// ```
/// use vecrem::VecExt;
///
/// let mut vec = vec![0, 1, 2, 3, 4];
/// core::mem::forget(vec.removing());
/// assert_eq!(vec, []);
/// ```
///
/// [`mem::forget`]: core::mem::forget
/// [`ManuallyDrop`]: core::mem::ManuallyDrop
pub struct Removing<'a, T> {
    // Type invariants:
    // - vec.capacity() >= len
    // - vec[..slot] is initialized (note: due to safety guarantees vec.len is set to 0, so this is
    //   not 'real' indexing)
    // - vec[curr..len] is initialized
    // - vec[..len] is the same allocation
    vec: &'a mut Vec<T>,
    slot: usize,
    curr: usize,
    len: usize,
}

/* Example of how the lib works, step-by-step:
 *
 * Imagine vec with items A-F:
 *
 * [A, B, C, D, E, F]
 *
 * 1. `Removing` is created
 *
 * [A, B, C, D, E, F]
 *  \
 *  curr, slot
 *
 * 2. `.next()` is used, entry is not removed
 *
 * [A, B, C, D, E, F]
 *     \
 *      curr, slot
 *
 * 3. `.next().remove()`
 *
 *       slot
 *      /
 * [A, _, C, D, E, F]
 *        \
 *         curr
 *
 * 4. `.next().remove()`
 *
 *       slot
 *      /
 * [A, _, _, D, E, F]
 *           \
 *            curr
 *
 * 5. `.next()`
 *
 *          slot
 *         /
 * [A, D, _, _, E, F]
 *              \
 *               curr
 *
 * 5. Removing::drop moves the rest of elements & restores `vec`'s len (in
 *    this case to 4):
 *
 * [A, D, E, F, _, _]
 */

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
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<Entry<'_, 'a, T>> {
        if self.is_empty() {
            None
        } else {
            // ## Safety
            //
            // Is not empty.
            Some(Entry { rem: self })
        }
    }

    /// Returns number of remaining elements in this pseudo-iterator
    ///
    /// ## Examples
    ///
    /// ```
    /// use vecrem::VecExt;
    ///
    /// let mut vec = vec![0, 1, 2, 3, 4];
    /// let mut rem = vec.removing();
    /// assert_eq!(rem.len(), 5);
    ///
    /// rem.next();
    /// assert_eq!(rem.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.len - self.curr
    }

    /// Return `true` if all elements of the underling vector were either
    /// ignored or [`remove`]d (i.e.: when [`next`] will return `None`)
    ///
    /// [`remove`]: Entry::remove
    /// [`next`]: Removing::next
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.curr == self.len
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
            let len = self.len();

            if len != 0 {
                // Copy the tail
                // [A, B, -, -, -, C, D, E, F] => [A, B, C, D, E, F, D*, E*, F*]
                // * logically this memory is uninitialized i.e. it's a move
                ptr::copy(self.curr_mut(), self.slot_mut(), len);
            }

            // Slot points to the first uninitialized element
            self.vec.set_len(self.slot + len)
        }
    }
}

/// Vec entry.
///
/// Entry allows you to [`remove`], [read] or [mutate] the element
/// behind it.
///
/// The only way to get this struct is to call [`Removing::next`] method.
///
/// [`remove`]: Entry::remove
/// [read]: Entry::value
/// [mutate]: Entry::value_mut
///
/// ## forget behavior
///
/// When [`Entry`] destructor (drop) is not runned (this can be achieved via
/// [`mem::forget`], [`ManuallyDrop`], etc) the entry is not skiped and the next
/// call to [`Removing::next`] returns the same entry.
///
/// ```
/// use core::mem;
/// use vecrem::VecExt;
///
/// let mut vec = vec![1, 2, 3];
/// {
///     let mut rem = vec.removing();
///
///     let a = rem.next().unwrap();
///     assert_eq!(a.value(), &1);
///     mem::forget(a);
///
///     let b = rem.next().unwrap();
///     assert_eq!(b.value(), &1);
///     mem::forget(b);
/// }
/// assert_eq!(vec, [1, 2, 3]);
/// ```
///
/// [`mem::forget`]: core::mem::forget
/// [`ManuallyDrop`]: core::mem::ManuallyDrop
pub struct Entry<'a, 'rem, T> {
    // Type invariants: !self.rem.is_empty()
    rem: &'a mut Removing<'rem, T>,
}

impl<T> Entry<'_, '_, T> {
    /// Remove element behind this entry.
    #[inline]
    pub fn remove(self) -> T {
        // Prevents `self` from dropping (Self::drop would skip element)
        let mut this = ManuallyDrop::new(self);

        unsafe {
            let curr = this.curr_mut();
            this.rem.curr += 1;

            // This read logically uninitializes mem behind `curr` ptr, but we've just moved
            // it, so it's ok.
            //
            // ## Safety
            //
            // Pointer is valid for reads by `Removing` invatiants.
            ptr::read(curr)
        }
    }

    /// Get access to the element behind this entry.
    #[inline]
    pub fn value(&self) -> &T {
        unsafe {
            // ## Safety
            //
            //`Entry` type invariants ensure that `curr` ptr is valid.
            &*self.curr_ptr()
        }
    }

    /// Get unique access to the element behind this entry.
    #[inline]
    pub fn value_mut(&mut self) -> &mut T {
        unsafe {
            // ## Safety
            //
            //`Entry` type invariants ensure that `curr` ptr is valid.
            &mut *self.curr_mut()
        }
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

        unsafe {
            // ## Safety
            //
            // We've just checked bounds
            Some(&mut *self.rem.ptr_mut(next))
        }
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

impl<T> Drop for Entry<'_, '_, T> {
    fn drop(&mut self) {
        // Skips element swapping it with the first empty slot.

        unsafe {
            // ## Safety
            //
            // `Entry` type invariants ensure that `self.rem.curr` points to a valid value,
            // and `self.rem.slot` is writable.
            ptr::copy(self.curr_mut(), self.slot_mut(), 1);
            self.rem.slot += 1;
            self.rem.curr += 1;
        }
    }
}

/// Extension for [`Vec`] which adds [`removing`] method
///
/// [`Vec`]: alloc::vec::Vec
/// [`removing`]: VecExt::removing
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

    use core::{fmt::Debug, mem, ops::Rem};

    /// Returns non-copy type that can help miri detect safety bugs
    fn f(i: i32) -> impl Clone + Debug + Eq + PartialOrd<i32> + Rem<i32, Output = i32> {
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct NoCopy(i32);

        impl PartialEq<i32> for NoCopy {
            fn eq(&self, other: &i32) -> bool {
                self.0.eq(other)
            }
        }

        impl PartialOrd<i32> for NoCopy {
            fn partial_cmp(&self, other: &i32) -> Option<core::cmp::Ordering> {
                self.0.partial_cmp(other)
            }
        }

        impl Rem<i32> for NoCopy {
            type Output = i32;

            fn rem(self, rem: i32) -> i32 {
                self.0 % rem
            }
        }

        NoCopy(i)
    }

    fn zf(_i: i32) -> impl Debug + Eq {
        #[derive(Debug, PartialEq, Eq)]
        struct No;
        No
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

        while let Some(_entry) = rem.next() {}
    }

    #[test]
    fn forget_entry() {
        let mut vec = vec![f(0)];
        {
            let mut rem = vec.removing();
            let mut timeout = 0..100;

            while let (Some(entry), Some(_)) = (rem.next(), timeout.next()) {
                mem::forget(entry)
            }

            assert_eq!(timeout.len(), 0);
        }
        assert_eq!(vec, [f(0)]);
    }

    #[test]
    fn zforget_entry() {
        let mut vec = vec![zf(0)];
        {
            let mut rem = vec.removing();
            let mut timeout = 0..100;

            while let (Some(entry), Some(_)) = (rem.next(), timeout.next()) {
                mem::forget(entry)
            }

            assert_eq!(timeout.len(), 0);
        }
        assert_eq!(vec, [zf(0)]);
    }

    #[test]
    // leaks mem
    #[cfg(not(miri))]
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

        assert_eq!(vec, [0; 0]);
    }

    #[test]
    fn drop() {
        let mut vec: Vec<_> = (0..10).map(f).collect();
        mem::drop(vec.removing());
        assert_eq!(vec, (0..10).map(f).collect::<Vec<_>>());
    }

    #[test]
    fn even() {
        let mut vec: Vec<_> = (0..10).map(f).collect();
        let mut out = Vec::with_capacity(10);
        let mut rem = vec.removing();

        while let Some(entry) = rem.next() {
            if entry.value().clone() % 2 == 0 {
                out.push(entry.remove());
            }
        }

        mem::drop(rem);

        assert_eq!(out, [0, 2, 4, 6, 8]);
        assert_eq!(vec, [1, 3, 5, 7, 9]);
    }

    #[test]
    fn break_() {
        let mut vec: Vec<_> = (0..10).map(f).collect();
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
        let mut vec: Vec<_> = (0..3).map(f).collect();

        {
            let mut rem = vec.removing();
            rem.next().unwrap();
            assert_eq!(rem.next().unwrap().remove(), 1);
            rem.next().unwrap();
        }

        assert_eq!(vec, [0, 2]);
    }

    #[test]
    fn zst() {
        let mut vec: Vec<_> = (0..3).map(zf).collect();

        {
            let mut rem = vec.removing();
            rem.next().unwrap();
            rem.next().unwrap().remove();
            rem.next().unwrap();
        }

        assert_eq!(vec.len(), 2);
    }
}
