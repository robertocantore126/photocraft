//! An optional faster compositor for large whole-document renders.
//!
//! The CPU reference compositor is exact but slow on big layered documents with layer effects:
//! a 20 MP, 479-layer PSB took minutes for a save's merged image, a histogram or an adjustment
//! preview, while the GPU compositor renders it in seconds. The app registers its GPU compositor
//! here ([`set`]); [`render`] uses it for large renders and falls back to the CPU (`None`) for
//! small ones, documents it does not handle, nested calls and when none is registered (the web,
//! tests, tools). `PHOTOCRAFT_GPU_COMPOSE=0` turns it off.
use photocraft_doc::Document;
use photocraft_geom::Rect;
use std::cell::Cell;
use std::sync::{Arc, PoisonError, RwLock};

/// Renders `rect` of a document as straight RGBA (row-major), or `None` to use the CPU.
pub type Accelerator = dyn Fn(&Document, Rect) -> Option<Vec<[f32; 4]>> + Send + Sync;

static ACCEL: RwLock<Option<Arc<Accelerator>>> = RwLock::new(None);

/// Renders below this many pixels stay on the CPU: they are quick there, and the GPU round trip
/// (planning every layer, readback) would not pay off.
const MIN_PIXELS: u64 = 512 * 512;

thread_local! {
    /// Set while the accelerator runs on this thread: anything it composites on the CPU itself
    /// must not come back here.
    static INSIDE: Cell<bool> = const { Cell::new(false) };
}

/// Register (or with `None` remove) the accelerator.
pub fn set(accelerator: Option<Arc<Accelerator>>) {
    *ACCEL.write().unwrap_or_else(PoisonError::into_inner) = accelerator;
}

/// `rect` of `doc` from the accelerator, when one is registered and takes it.
pub(crate) fn render(doc: &Document, rect: Rect) -> Option<crate::Buffer> {
    if std::env::var("PHOTOCRAFT_GPU_COMPOSE").as_deref() == Ok("0") {
        return None;
    }
    let accelerator = ACCEL.read().unwrap_or_else(PoisonError::into_inner).clone()?;
    render_with(&*accelerator, doc, rect)
}

fn render_with(accelerator: &Accelerator, doc: &Document, rect: Rect) -> Option<crate::Buffer> {
    if rect.is_empty() || rect.width() as u64 * rect.height() as u64 <= MIN_PIXELS || INSIDE.with(Cell::get) {
        return None;
    }
    // RGB only: Lab mixes Normal blending in CIELAB, and the other modes have their own backdrop
    // and channel handling, which the CPU compositor implements.
    if doc.mode != photocraft_color::ColorMode::Rgb {
        return None;
    }
    INSIDE.with(|inside| inside.set(true));
    let px = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| accelerator(doc, rect))).ok().flatten();
    INSIDE.with(|inside| inside.set(false));
    let px = px.filter(|px| px.len() == rect.width() as usize * rect.height() as usize)?;
    Some(crate::Buffer { rect, px })
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::SampleType;
    use photocraft_doc::{ColorMode, Size};

    #[test]
    fn large_renders_use_the_accelerator_and_small_or_nested_ones_do_not() {
        let doc = Document::new("a", Size::new(1024, 1024), ColorMode::Rgb, SampleType::U8);
        let fill = |d: &Document, r: Rect| {
            // A nested render inside the accelerator stays on the CPU.
            let nested = render_with(&|_: &Document, _: Rect| Some(Vec::new()), d, r);
            assert!(nested.is_none());
            Some(vec![[0.25, 0.5, 0.75, 1.0]; r.width() as usize * r.height() as usize])
        };
        assert_eq!(render_with(&fill, &doc, doc.bounds()).map(|b| b.px[0]), Some([0.25, 0.5, 0.75, 1.0]));
        assert!(render_with(&fill, &doc, Rect::new(0, 0, 64, 64)).is_none());
        assert!(render_with(&|_: &Document, _: Rect| Some(vec![[0.0; 4]; 3]), &doc, doc.bounds()).is_none());
        let lab = Document::new("l", Size::new(1024, 1024), ColorMode::Lab, SampleType::U8);
        assert!(render_with(&fill, &lab, lab.bounds()).is_none());
    }
}
