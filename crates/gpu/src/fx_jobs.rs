//! Bounded, native-only CPU effect inputs. GPU uploads stay on the render thread.
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, mpsc};

use photocraft_compose::effects::FieldKind;
use photocraft_doc::{DocId, Document, Layer};
use photocraft_geom::Rect;

use crate::{FxKey, fx};

pub(crate) struct Inputs {
    pub shape: Vec<f32>,
    pub fields: HashMap<FieldKind, (i32, Vec<f32>)>,
}

struct Request {
    key: (DocId, FxKey),
    serial: u64,
    doc: Document,
    layer: Layer,
    region: Rect,
    fields: HashMap<FieldKind, i32>,
}

struct Reply {
    key: (DocId, FxKey),
    serial: u64,
    region: Rect,
    inputs: Result<Inputs, String>,
}

struct Pending {
    signature: u64,
    bytes: u64,
    serial: u64,
    reply: Option<Result<Inputs, String>>,
    /// Render (`Jobs::next_frame`) that last asked for it.
    last_polled: u64,
}

/// Requests waiting in the channel plus one running on each worker: a full channel never
/// rejects a request this admits, so no document snapshot is made only to be dropped.
const QUEUE: usize = 8;
const WORKERS: usize = 2;
const IN_FLIGHT: usize = QUEUE + WORKERS;
/// Bytes of queued and completed inputs held at once.
const RESERVE: u64 = 256 << 20;
/// A completed result its layer has not asked for in this many renders may be dropped under
/// memory pressure; after `STALE` renders it is dropped regardless.
const IDLE: u64 = 8;
const STALE: u64 = 120;

pub(crate) enum Poll {
    Waiting,
    Ready(Inputs),
    Failed(String),
}

pub(crate) struct Jobs {
    sender: Option<mpsc::SyncSender<Request>>,
    receiver: Mutex<mpsc::Receiver<Reply>>,
    stopped: Arc<AtomicBool>,
    pending: HashMap<(DocId, FxKey), Pending>,
    damage: HashMap<DocId, Rect>,
    serial: u64,
    /// Top-level renders so far (see `next_frame`).
    frame: u64,
}

impl Jobs {
    pub fn new() -> Result<Self, std::io::Error> {
        let (sender, requests) = mpsc::sync_channel::<Request>(QUEUE);
        let (replies, receiver) = mpsc::sync_channel(2);
        let requests = Arc::new(Mutex::new(requests));
        let stopped = Arc::new(AtomicBool::new(false));
        for worker in 0..WORKERS {
            let requests = requests.clone();
            let replies = replies.clone();
            let stop = stopped.clone();
            if let Err(error) = std::thread::Builder::new().name(format!("photocraft-effects-{worker}")).spawn(move || {
                loop {
                    let request = requests.lock().unwrap_or_else(PoisonError::into_inner).recv();
                    let Ok(request) = request else { break };
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    let span = photocraft_trace::span_with("fx_inputs", || {
                        format!("{} ({}x{}, {} fields)", request.layer.name, request.region.width(), request.region.height(), request.fields.len())
                    });
                    let inputs = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let shape = fx::shape(&request.doc, &request.layer, request.region);
                        let mut fields = HashMap::new();
                        for (kind, reach) in request.fields {
                            if stop.load(Ordering::Acquire) {
                                return Err("effect worker stopped".into());
                            }
                            let values = fx::field(kind, reach, &shape, request.region, request.region);
                            fields.insert(kind, (reach, values));
                        }
                        Ok(Inputs { shape, fields })
                    }))
                    .unwrap_or_else(|_| Err("effect input computation failed with an internal error".into()));
                    drop(span);
                    let drop_span = photocraft_trace::span("fx_request_drop");
                    let Request { key, serial, region, .. } = request;
                    drop(drop_span);
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    if replies.send(Reply { key, serial, region, inputs }).is_err() {
                        break;
                    }
                }
            }) {
                stopped.store(true, Ordering::Release);
                return Err(error);
            }
        }
        Ok(Self { sender: Some(sender), receiver: Mutex::new(receiver), stopped, pending: HashMap::new(), damage: HashMap::new(), serial: 0, frame: 0 })
    }

    /// A top-level render begins: results nobody has asked for in a long while are dropped.
    pub fn next_frame(&mut self) {
        self.frame += 1;
        let frame = self.frame;
        self.pending.retain(|_, p| p.reply.is_none() || p.last_polled + STALE >= frame);
    }

    pub fn collect(&mut self) {
        while let Ok(reply) = self.receiver.get_mut().unwrap_or_else(PoisonError::into_inner).try_recv() {
            let Some(pending) = self.pending.get_mut(&reply.key).filter(|p| p.serial == reply.serial) else { continue };
            pending.reply = Some(reply.inputs);
            self.damage.entry(reply.key.0).and_modify(|r| *r = r.union(&reply.region)).or_insert(reply.region);
        }
    }

    pub fn poll(&mut self, key: FxKey, signature: u64, doc: &Document, layer: &Layer, region: Rect, fields: &HashMap<FieldKind, i32>) -> Poll {
        self.collect();
        let key = (doc.id, key);
        if self.pending.get(&key).is_some_and(|p| p.signature != signature && p.reply.is_none()) {
            photocraft_trace::instant("fx_wait_superseded", || Some(layer.name.clone()));
            return Poll::Waiting;
        }
        if self.pending.get(&key).is_some_and(|p| p.signature != signature) {
            photocraft_trace::instant("fx_discarded", || Some(layer.name.clone()));
            self.pending.remove(&key);
        }
        if let Some(pending) = self.pending.get_mut(&key).filter(|p| p.signature == signature) {
            pending.last_polled = self.frame;
            return match pending.reply.take() {
                Some(Ok(inputs)) => {
                    photocraft_trace::instant("fx_ready", || Some(layer.name.clone()));
                    self.pending.remove(&key);
                    Poll::Ready(inputs)
                }
                Some(Err(error)) => {
                    photocraft_trace::instant("fx_failed", || Some(format!("{}: {error}", layer.name)));
                    self.pending.remove(&key);
                    Poll::Failed(error)
                }
                None => Poll::Waiting,
            };
        }
        // Bound snapshots as well as queued and completed allocations. Retry next frame
        // when the queue is full rather than doing the expensive work on the UI thread.
        // Only work in flight counts against the queue: a completed result waits for its layer
        // to collect it later in this same render. Dropping completed results to make room
        // (as before) livelocked as soon as more layers had effects than there were slots:
        // every result was discarded before its layer asked, so every layer re-queued forever.
        if self.pending.values().filter(|p| p.reply.is_none()).count() >= IN_FLIGHT {
            photocraft_trace::instant("fx_deferred", || Some(format!("{}: {IN_FLIGHT} in flight", layer.name)));
            self.damage.entry(doc.id).and_modify(|r| *r = r.union(&region)).or_insert(region);
            return Poll::Waiting;
        }
        let bytes = u64::from(region.width())
            .checked_mul(u64::from(region.height()))
            .and_then(|pixels| pixels.checked_mul((fields.len() as u64).saturating_add(1)))
            .and_then(|samples| samples.checked_mul(4));
        let Some(bytes) = bytes.filter(|bytes| *bytes <= RESERVE) else {
            return Poll::Failed("effect input window exceeds the worker memory limit".into());
        };
        let mut held = self.pending.values().fold(0u64, |sum, pending| sum.saturating_add(pending.bytes));
        if held.saturating_add(bytes) > RESERVE {
            // Free completed results whose layers stopped asking (offscreen), never the ones
            // still being collected.
            let frame = self.frame;
            self.pending.retain(|_, p| p.reply.is_none() || p.last_polled + IDLE >= frame);
            held = self.pending.values().fold(0u64, |sum, pending| sum.saturating_add(pending.bytes));
        }
        if held.saturating_add(bytes) > RESERVE {
            photocraft_trace::instant("fx_deferred", || Some(format!("{}: 256 MiB reserve", layer.name)));
            self.damage.entry(doc.id).and_modify(|r| *r = r.union(&region)).or_insert(region);
            return Poll::Waiting;
        }
        self.serial = self.serial.wrapping_add(1);
        let serial = self.serial;
        let clone = photocraft_trace::span_with("fx_request_clone", || format!("{} layers", doc.layers.len()));
        let request = Request { key, serial, doc: doc.clone(), layer: layer.clone(), region, fields: fields.clone() };
        drop(clone);
        let Some(sender) = &self.sender else { return Poll::Failed("effect worker disconnected".into()) };
        match sender.try_send(request) {
            Ok(()) => {
                photocraft_trace::instant("fx_queued", || Some(format!("{} ({}x{})", layer.name, region.width(), region.height())));
                self.pending.insert(key, Pending { signature, bytes, serial, reply: None, last_polled: self.frame });
                Poll::Waiting
            }
            Err(mpsc::TrySendError::Full(rejected)) => {
                photocraft_trace::instant("fx_deferred", || Some(format!("{}: queue full", layer.name)));
                let drop_span = photocraft_trace::span("fx_request_drop");
                drop(rejected);
                drop(drop_span);
                self.damage.entry(doc.id).and_modify(|r| *r = r.union(&region)).or_insert(region);
                Poll::Waiting
            }
            Err(mpsc::TrySendError::Disconnected(_)) => Poll::Failed("effect worker disconnected".into()),
        }
    }

    /// Work in flight for `doc`, or a request deferred because the queue or the memory reserve
    /// was full: its damage is retried on the next frame, so the caller must ask for one.
    pub fn pending(&self, doc: DocId) -> bool {
        // A completed result counts while its layer still asks for it: one more render
        // collects it. One nobody asked for lately (an offscreen layer) does not keep frames
        // coming.
        let frame = self.frame;
        self.pending.iter().any(|((id, _), p)| *id == doc && (p.reply.is_none() || p.last_polled + IDLE >= frame)) || self.damage.contains_key(&doc)
    }

    pub fn take_damage(&mut self, doc: DocId) -> Option<Rect> {
        self.collect();
        self.damage.remove(&doc)
    }

    pub fn forget_doc(&mut self, doc: DocId) {
        self.pending.retain(|(id, _), _| *id != doc);
        self.damage.remove(&doc);
    }
}

impl Drop for Jobs {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);
        self.sender.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::{ColorMode, LayerId, SampleType, Size};

    /// BUG-04: a tab whose request is deferred (every reservation held by another tab) must still
    /// report pending work, or the canvas stops asking for frames and its effects never appear.
    #[test]
    fn deferred_request_keeps_the_document_pending() {
        let mut jobs = Jobs::new().unwrap();
        let a = Document::new("a", Size::new(64, 64), ColorMode::Rgb, SampleType::U8);
        let b = Document::new("b", Size::new(64, 64), ColorMode::Rgb, SampleType::U8);
        for i in 0..IN_FLIGHT as u64 {
            jobs.pending.insert((a.id, (LayerId(i), None, (64, 64))), Pending { signature: 1, bytes: 0, serial: i, reply: None, last_polled: 0 });
        }
        let layer = photocraft_doc::Layer::raster("l", b.pixel_format());
        let region = Rect::new(0, 0, 64, 64);
        assert!(matches!(jobs.poll((layer.id, None, (64, 64)), 7, &b, &layer, region, &HashMap::new()), Poll::Waiting));
        assert!(!jobs.pending.keys().any(|(id, _)| *id == b.id), "the request was deferred, not queued");
        assert!(jobs.pending(b.id), "a deferred request must keep the canvas repainting");
        // The next frame turns the damage into dirty cells and polls again.
        assert_eq!(jobs.take_damage(b.id), Some(region));
        assert!(!jobs.pending(b.id));
    }

    /// With more effect layers than queue slots, a result completed for one layer must survive
    /// another layer's request in the same render until its own layer collects it. Dropping
    /// completed results to make room livelocked a 479-layer PSB: every layer re-queued its
    /// inputs every frame and none ever arrived (thousands queued, zero ready, in one minute).
    #[test]
    fn completed_results_survive_until_collected() {
        let mut jobs = Jobs::new().unwrap();
        let doc = Document::new("d", Size::new(64, 64), ColorMode::Rgb, SampleType::U8);
        let layer = photocraft_doc::Layer::raster("l", doc.pixel_format());
        let region = Rect::new(0, 0, 64, 64);
        jobs.next_frame();
        let done = |serial| Pending {
            signature: 7,
            bytes: 64 * 64 * 4,
            serial,
            reply: Some(Ok(Inputs { shape: vec![0.0; 64 * 64], fields: HashMap::new() })),
            last_polled: 1,
        };
        for i in 0..20u64 {
            jobs.pending.insert((doc.id, (LayerId(100 + i), None, (64, 64))), done(i));
        }
        // Another layer asks first in this render: it is admitted (nothing is in flight) and
        // must not cost the twenty completed results their place.
        let other = (LayerId(1), None, (64, 64));
        assert!(matches!(jobs.poll(other, 9, &doc, &layer, region, &HashMap::new()), Poll::Waiting));
        for i in 0..20u64 {
            let key = (LayerId(100 + i), None, (64, 64));
            assert!(matches!(jobs.poll(key, 7, &doc, &layer, region, &HashMap::new()), Poll::Ready(_)), "layer {i} lost its result");
        }
        // Results nobody collects go away eventually and stop counting as pending work.
        jobs.pending.insert((doc.id, (LayerId(999), None, (64, 64))), done(99));
        for _ in 0..=STALE {
            jobs.next_frame();
        }
        assert!(!jobs.pending.contains_key(&(doc.id, (LayerId(999), None, (64, 64)))));
    }
}
