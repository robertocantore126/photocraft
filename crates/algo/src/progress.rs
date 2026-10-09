//! Cooperative cancellation shared by tile algorithms and their callers.
use std::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU32, Ordering},
};

#[derive(Clone, Default)]
pub struct Progress(Arc<State>);
#[derive(Default)]
struct State {
    fraction: AtomicU32,
    cancelled: AtomicBool,
}
impl Progress {
    pub fn set_fraction(&self, fraction: f32) {
        self.0.fraction.store(fraction.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }
    pub fn fraction(&self) -> f32 {
        f32::from_bits(self.0.fraction.load(Ordering::Relaxed))
    }
    pub fn cancel(&self) {
        self.0.cancelled.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::Relaxed)
    }
}
thread_local! { static CURRENT: RefCell<Option<Progress>> = const { RefCell::new(None) }; }
pub fn current() -> Option<Progress> {
    CURRENT.with(|slot| slot.borrow().clone())
}
/// Install the progress handle for synchronous command code, including nested filters.
/// Algorithms capture it before entering Rayon, so worker threads need no thread local state.
pub fn with_progress<R>(progress: &Progress, run: impl FnOnce() -> R) -> R {
    struct Restore(Option<Progress>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let old = CURRENT.with(|slot| slot.replace(Some(progress.clone())));
    let _restore = Restore(old);
    run()
}
