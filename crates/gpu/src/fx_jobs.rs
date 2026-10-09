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
}

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
}

impl Jobs {
    pub fn new() -> Result<Self, std::io::Error> {
        let (sender, requests) = mpsc::sync_channel::<Request>(8);
        let (replies, receiver) = mpsc::sync_channel(2);
        let requests = Arc::new(Mutex::new(requests));
        let stopped = Arc::new(AtomicBool::new(false));
        for worker in 0..2 {
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
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    if replies.send(Reply { key: request.key, serial: request.serial, region: request.region, inputs }).is_err() {
                        break;
                    }
                }
            }) {
                stopped.store(true, Ordering::Release);
                return Err(error);
            }
        }
        Ok(Self { sender: Some(sender), receiver: Mutex::new(receiver), stopped, pending: HashMap::new(), damage: HashMap::new(), serial: 0 })
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
            return Poll::Waiting;
        }
        if self.pending.get(&key).is_some_and(|p| p.signature != signature) {
            self.pending.remove(&key);
        }
        if let Some(pending) = self.pending.get_mut(&key).filter(|p| p.signature == signature) {
            return match pending.reply.take() {
                Some(Ok(inputs)) => {
                    self.pending.remove(&key);
                    Poll::Ready(inputs)
                }
                Some(Err(error)) => {
                    self.pending.remove(&key);
                    Poll::Failed(error)
                }
                None => Poll::Waiting,
            };
        }
        // Bound snapshots as well as queued and completed allocations. Retry next frame
        // when the queue is full rather than doing the expensive work on the UI thread.
        // Completed offscreen requests must not permanently reserve every slot. They
        // can be rebuilt if their layer becomes visible again; in-flight work stays tracked.
        if self.pending.len() >= 12 {
            self.pending.retain(|_, pending| pending.reply.is_none());
        }
        if self.pending.len() >= 12 && !self.pending.contains_key(&key) {
            self.damage.entry(doc.id).and_modify(|r| *r = r.union(&region)).or_insert(region);
            return Poll::Waiting;
        }
        let bytes = u64::from(region.width())
            .checked_mul(u64::from(region.height()))
            .and_then(|pixels| pixels.checked_mul((fields.len() as u64).saturating_add(1)))
            .and_then(|samples| samples.checked_mul(4));
        let Some(bytes) = bytes.filter(|bytes| *bytes <= 256 << 20) else {
            return Poll::Failed("effect input window exceeds the worker memory limit".into());
        };
        let mut held = self.pending.values().fold(0u64, |sum, pending| sum.saturating_add(pending.bytes));
        if held.saturating_add(bytes) > 256 << 20 {
            self.pending.retain(|_, pending| pending.reply.is_none());
            held = self.pending.values().fold(0u64, |sum, pending| sum.saturating_add(pending.bytes));
        }
        if held.saturating_add(bytes) > 256 << 20 {
            self.damage.entry(doc.id).and_modify(|r| *r = r.union(&region)).or_insert(region);
            return Poll::Waiting;
        }
        self.serial = self.serial.wrapping_add(1);
        let serial = self.serial;
        let request = Request { key, serial, doc: doc.clone(), layer: layer.clone(), region, fields: fields.clone() };
        match self.sender.as_ref().map(|s| s.try_send(request)) {
            Some(Ok(())) => {
                self.pending.insert(key, Pending { signature, bytes, serial, reply: None });
                Poll::Waiting
            }
            Some(Err(mpsc::TrySendError::Full(_))) => {
                self.damage.entry(doc.id).and_modify(|r| *r = r.union(&region)).or_insert(region);
                Poll::Waiting
            }
            _ => Poll::Failed("effect worker disconnected".into()),
        }
    }

    pub fn pending(&self, doc: DocId) -> bool {
        self.pending.keys().any(|(id, _)| *id == doc)
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
