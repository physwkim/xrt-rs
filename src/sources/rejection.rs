//! Termination guard for the Monte Carlo rejection loops.
//!
//! `shine` on every synchrotron source repeats "sample a batch, keep the rays
//! that pass the intensity discriminator" until it has `nrays`. A batch that
//! yields nothing is normal — Python xrt prints `'No good rays in this seed!'`
//! and retries (sources/synchr.py:363-369). What Python has no answer for is a
//! configuration that can never yield a ray, such as an energy band far above
//! the critical energy: there the loop spins forever, silently.
//!
//! The invariant this module owns: **a `shine` loop must either make progress
//! or stop.** Every path that abandons a batch goes through
//! [`RejectionBudget::note_empty_batch`], every path that keeps rays goes
//! through [`RejectionBudget::note_progress`], and no loop reads the counter
//! itself.

/// Consecutive empty batches tolerated before a loop is declared stuck.
///
/// A batch is `1.2 * nrays` samples, so this is a hard limit of ~1e3 batches
/// of futile sampling — orders of magnitude more than a healthy source needs
/// (the worst legitimate case measured here accepts one ray in 5e4 samples),
/// and small enough to fail in seconds rather than hours.
const MAX_EMPTY_BATCHES: usize = 1000;

/// Counts consecutive fruitless sampling batches for one `shine` call.
pub(crate) struct RejectionBudget {
    consecutive_empty: usize,
}

impl RejectionBudget {
    pub(crate) fn new() -> Self {
        Self {
            consecutive_empty: 0,
        }
    }

    /// Record a batch that produced no rays.
    ///
    /// Panics once [`MAX_EMPTY_BATCHES`] batches in a row have produced
    /// nothing, with `diagnostic` naming the parameters that make the source
    /// unable to emit. `diagnostic` is only called on that failure.
    pub(crate) fn note_empty_batch(&mut self, source: &str, diagnostic: impl FnOnce() -> String) {
        self.consecutive_empty += 1;
        if self.consecutive_empty >= MAX_EMPTY_BATCHES {
            panic!(
                "{source}::shine made no progress in {MAX_EMPTY_BATCHES} consecutive sampling \
                 batches: {}",
                diagnostic()
            );
        }
    }

    /// Record a batch that produced at least one ray.
    pub(crate) fn note_progress(&mut self) {
        self.consecutive_empty = 0;
    }
}
