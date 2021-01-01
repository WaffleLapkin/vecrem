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
use core::{mem::ManuallyDrop, ptr, ptr::NonNull};

// FIXME(waffle): add support for backwards iteration

/// A cursor-like structure made for cheap iterative removing of elements from
/// vector.
///
/// Essentially overhead of the process is one `ptr::copy` for each ignored
/// element element + one `ptr::copy` for the not yielded tail.
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
/// ## Leak Notes
///
/// In the same way as [`Vec::drain_filter`] for efishient work, [`Removing`]
/// needs to remporary break vec leaving it in incunsisntent state.
/// The state is made normal in `Drop`, however in rust running destructors is
/// not guaranteed (see [`mem::forget`]). As such, [`Removing`] sets vec's len
/// to `0` (and restores it in `Drop`), this means that if [`Removing`] gets
/// leaked or forgotten - the elements of the vectore are gone too.
///
/// [`mem::forget`]: core::mem::forget
pub struct Removing<'a, T> {
    // Type invariants:
    // - `vec`'s len set to 0, so it's safe to do anything with it's buf (it's crucial for
    //   forget-safety, similar to Vec::drain behaviour)
    // - `slot` is pointing into vec's buffer
    // - `curr` and `end` are pointing into vec's buffer or 1 item past vec's buffer
    vec: &'a mut Vec<T>,

    // Derived from `vec`
    /// Points to the start of the buffer
    #[cfg(all(x, not(x)))]
    _start: NonNull<T>,

    /// Points to the first empty 'slot'. If no items were removed, then points
    /// to the same item as `curr`. Otherwise pints to the first empty
    /// place.
    slot: NonNull<T>,

    /// Points to the next item that will be yilded.
    curr: NonNull<T>,

    /// Points one element past the last element
    end: NonNull<T>,
    /* Example of how this works, step-by-step:
     *
     * Imagine vec with items A-F:
     *
     * [A, B, C, D, E, F]
     *
     * 1. `Removing` is created
     *
     * [A, B, C, D, E, F]
     *  \                ^-- end
     *  start, curr, slot
     *
     * 2. `.next()` is used, entry is not removed
     *
     *       curr, slot
     *      /
     * [A, B, C, D, E, F]
     *  \                ^-- end
     *  start
     *
     * 3. `.next().remove()`
     *
     *       slot
     *      /
     * [A, _, C, D, E, F]
     *  \     \          ^-- end
     * start   curr
     *
     * 4. `.next().remove()`
     *
     *       slot
     *      /
     * [A, _, _, D, E, F]
     *  \        \       ^-- end
     * start      curr
     *
     * 5. `.next()`
     *
     *          slot
     *         /
     * [A, D, _, _, E, F]
     *  \           \    ^-- end
     * start         curr
     *
     * 5. Removing::drop moves the rest of elements & restores `vec`'s len (in this case to 4):
     *
     * [A, D, E, F, _, _]
     */
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

        let start = unsafe {
            let ptr = vec.as_mut_ptr();
            debug_assert!(!ptr.is_null());
            NonNull::new_unchecked(ptr)
        };

        Self {
            vec,
            slot: start,
            curr: start,
            end: unsafe { NonNull::new_unchecked(start.as_ptr().add(len)) },
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
        unsafe {
            debug_assert!(self.curr.as_ptr() as usize <= self.end.as_ptr() as usize);

            // ## Safety
            //
            // Both `curr` and `end` are pointing into vec's buffer/one element past it.
            //
            // `curr <= end`
            self.end.as_ptr().offset_from(self.curr.as_ptr()) as _
        }
    }

    /// Return `true` if all elements of the underling vector were either
    /// ignored or [`remove`]d (i.e.: when [`next`] will return `None`)
    ///
    /// [`remove`]: Entry::remove
    /// [`next`]: Removing::next
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.curr == self.end
    }

    fn start(&mut self) -> NonNull<T> {
        unsafe {
            // ## Safety
            //
            // vec's ptr is never null
            NonNull::new_unchecked(self.vec.as_mut_ptr())
        }
    }
}

impl<T> Drop for Removing<'_, T> {
    fn drop(&mut self) {
        unsafe {
            let tail_len = self.len();

            // Copy the tail
            // [A, B, -, -, -, C, D, E, F] => [A, B, C, D, E, F, D*, E*, F*]
            // * logically this memory is uninitialized i.e. it's a move
            ptr::copy(self.curr.as_ptr(), self.slot.as_ptr(), tail_len);

            let head_len = self.slot.as_ptr().offset_from(self.start().as_ptr()) as usize;

            // Slot points to the first uninitialized element
            self.vec.set_len(head_len + tail_len);
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
            // ## Safety
            //
            // `curr` points into initialized element by type invariants
            let ret = ptr::read(this.rem.curr.as_ptr());

            // ## Safety
            //
            // `curr` points into Vec's buffer, so +1 can't be UB
            this.rem.curr = add(this.rem.curr, 1);
            ret
        }
    }

    /// Get access to the element behind this entry.
    #[inline]
    pub fn value(&self) -> &T {
        unsafe {
            // ## Safety
            //
            // `curr` points into initialized element by type invariants
            self.rem.curr.as_ref()
        }
    }

    /// Get unique access to the element behind this entry.
    #[inline]
    pub fn value_mut(&mut self) -> &mut T {
        unsafe {
            // ## Safety
            //
            // `curr` points into initialized element by type invariants
            self.rem.curr.as_mut()
        }
    }

    /// Get unique access to the element after the element behind this entry.
    ///
    /// Returns `None` if this entry corresponds to the last item in the vec.
    #[inline]
    pub fn peek_next(&mut self) -> Option<&mut T> {
        if self.rem.is_empty() {
            return None;
        }

        let next = unsafe {
            // ## Safety
            //
            // `curr` points into Vec's buffer, so +1 can't be UB
            add(self.rem.curr, 1)
        };

        if next == self.rem.end {
            return None;
        }

        unsafe {
            // ## Safety
            //
            // `curr` points into vec's buffer & into initialized element
            Some(&mut *next.as_ptr())
        }
    }
}

impl<T> Drop for Entry<'_, '_, T> {
    fn drop(&mut self) {
        // Skips element swaping it with the last empty slot.

        unsafe {
            // ## Safety
            //
            // `Entry` type invariants ensure that `self.rem.curr` points to a valid value,
            // and `self.rem.slot` is writable.
            self.rem.slot.as_ptr().copy_from(self.rem.curr.as_ptr(), 1);

            // ## Safety
            //
            // `Entry` type invariants ensure that both slot `slot` and `curr` are pointing
            // into the vec buffer, so +1 may not cause UB
            self.rem.slot = add(self.rem.slot, 1);
            self.rem.curr = add(self.rem.curr, 1);
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

/// Safety: same as <*mut T>::add
unsafe fn add<T>(ptr: NonNull<T>, count: usize) -> NonNull<T> {
    NonNull::new_unchecked(ptr.as_ptr().add(count))
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

        while let Some(_entry) = rem.next() {}
    }

    #[test]
    fn forget_entry() {
        let mut vec = vec![f(0)];
        let mut rem = vec.removing();
        let mut timeout = 0..100;

        while let (Some(entry), Some(_)) = (rem.next(), timeout.next()) {
            mem::forget(entry)
        }

        assert_eq!(timeout.len(), 0);
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
            rem.next().unwrap();
            assert_eq!(rem.next().unwrap().remove(), 1);
            rem.next().unwrap();
        }

        assert_eq!(vec, [0, 2]);
    }
}
