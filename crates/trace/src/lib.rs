//! In-process action tracer.
//!
//! Spans (timed sections), user actions, instant events and counters go into a ring buffer that
//! keeps the last [`WINDOW_SECONDS`]. [`save`] writes it as one JSON file that is both a Chrome
//! trace (`traceEvents`, opens in ui.perfetto.dev) and a per-action summary (`photocraft`): for
//! each action, the first frame after it, when the canvas settled, frame times, where the main
//! thread spent the time and what background threads did.
//!
//! On by default on native builds (`PHOTOCRAFT_TRACE=0` turns it off); never on the web.
//! Recording costs one short mutex push per event, so spans belong around whole phases (a frame,
//! a composite, one layer's effect maps), never per tile or per pixel.
use serde_json::{Map, Value, json};
use std::borrow::Cow;
use std::cell::Cell;
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Instant;

/// Seconds of history kept (and saved).
pub const WINDOW_SECONDS: u64 = 60;
/// Hard cap on buffered events, whatever their age.
const MAX_EVENTS: usize = 2_000_000;
/// Counter the canvas reports at the end of each frame: 1 while it still has work to show.
pub const CANVAS_PENDING: &str = "canvas_pending";
/// Span the UI wraps around each frame on the main thread.
pub const FRAME: &str = "frame";

#[derive(Clone, Debug)]
enum Kind {
    Span(u64),
    Action,
    Instant,
    Counter(f64),
}

#[derive(Clone, Debug)]
struct Event {
    kind: Kind,
    name: Cow<'static, str>,
    tid: u32,
    /// Microseconds since the tracer started.
    ts: u64,
    arg: Option<String>,
}

struct Store {
    events: VecDeque<Event>,
    threads: BTreeMap<u32, String>,
}

static STORE: Mutex<Store> = Mutex::new(Store { events: VecDeque::new(), threads: BTreeMap::new() });
static NEXT_TID: AtomicU32 = AtomicU32::new(1);
thread_local! {
    static TID: Cell<u32> = const { Cell::new(0) };
}

/// Whether events are recorded.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(not(target_arch = "wasm32")) && std::env::var("PHOTOCRAFT_TRACE").as_deref() != Ok("0"))
}

fn start() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

fn micros(at: Instant) -> u64 {
    u64::try_from(at.saturating_duration_since(start()).as_micros()).unwrap_or(u64::MAX)
}

fn tid() -> u32 {
    TID.with(|t| {
        if t.get() == 0 {
            let id = NEXT_TID.fetch_add(1, Ordering::Relaxed);
            t.set(id);
            let name = std::thread::current().name().map_or_else(|| format!("worker-{id}"), str::to_owned);
            STORE.lock().unwrap_or_else(PoisonError::into_inner).threads.insert(id, name);
        }
        t.get()
    })
}

fn push(kind: Kind, name: Cow<'static, str>, ts: u64, arg: Option<String>) {
    let tid = tid();
    let mut store = STORE.lock().unwrap_or_else(PoisonError::into_inner);
    let horizon = ts.saturating_sub(WINDOW_SECONDS * 1_000_000);
    while store.events.front().is_some_and(|e| e.ts < horizon) || store.events.len() >= MAX_EVENTS {
        store.events.pop_front();
    }
    store.events.push_back(Event { kind, name, tid, ts, arg });
}

/// A timed section, recorded when dropped.
#[must_use = "a span measures until it is dropped"]
pub struct Span {
    name: &'static str,
    arg: Option<String>,
    start: Option<Instant>,
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(begin) = self.start {
            let ts = micros(begin);
            let dur = u64::try_from(begin.elapsed().as_micros()).unwrap_or(u64::MAX);
            push(Kind::Span(dur), Cow::Borrowed(self.name), ts, self.arg.take());
        }
    }
}

/// Time a section until the returned guard drops.
pub fn span(name: &'static str) -> Span {
    Span { name, arg: None, start: enabled().then(Instant::now) }
}

/// [`span`] with a detail (layer, command id, size); `arg` runs only while tracing.
pub fn span_with(name: &'static str, arg: impl FnOnce() -> String) -> Span {
    let on = enabled();
    Span { name, arg: on.then(arg), start: on.then(Instant::now) }
}

/// A user action (a command, a stroke, a zoom): the summary groups everything after it.
pub fn action(name: impl FnOnce() -> String) {
    if enabled() {
        push(Kind::Action, Cow::Owned(name()), micros(Instant::now()), None);
    }
}

/// A point event (a job queued, deferred, discarded) with an optional detail.
pub fn instant(name: &'static str, arg: impl FnOnce() -> Option<String>) {
    if enabled() {
        push(Kind::Instant, Cow::Borrowed(name), micros(Instant::now()), arg());
    }
}

/// A sampled value (pending work, queue length).
pub fn counter(name: &'static str, value: f64) {
    if enabled() {
        push(Kind::Counter(value), Cow::Borrowed(name), micros(Instant::now()), None);
    }
}

/// Write the buffered window to `dir` as `trace-<UTC date>-<time>.json`; returns its path.
pub fn save(dir: &Path) -> std::io::Result<PathBuf> {
    let (events, threads) = {
        let store = STORE.lock().unwrap_or_else(PoisonError::into_inner);
        (store.events.iter().cloned().collect::<Vec<_>>(), store.threads.clone())
    };
    let doc = to_json(&events, &threads);
    std::fs::create_dir_all(dir)?;
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let path = dir.join(format!("trace-{}.json", utc_stamp(secs)));
    let text = serde_json::to_string(&doc).map_err(std::io::Error::other)?;
    std::fs::write(&path, text)?;
    Ok(path)
}

/// `YYYYMMDD-HHMMSS` (UTC) for Unix seconds.
fn utc_stamp(secs: u64) -> String {
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let rem = secs % 86_400;
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}{month:02}{day:02}-{:02}{:02}{:02}", rem / 3600, rem / 60 % 60, rem % 60)
}

fn ms(us: u64) -> f64 {
    (us as f64 / 1000.0 * 10.0).round() / 10.0
}

fn stats(mut v: Vec<f64>) -> Value {
    if v.is_empty() {
        return Value::Null;
    }
    v.sort_by(f64::total_cmp);
    let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    json!({ "p50": at(0.5), "p95": at(0.95), "max": at(1.0) })
}

fn top(map: BTreeMap<String, u64>, n: usize) -> Value {
    let mut v: Vec<(String, u64)> = map.into_iter().filter(|(_, us)| *us >= 500).collect();
    v.sort_by_key(|e| std::cmp::Reverse(e.1));
    Value::Object(v.into_iter().take(n).map(|(k, us)| (k, json!(ms(us)))).collect::<Map<_, _>>())
}

/// Microseconds of `[s, s + d)` inside `[a, b)`.
fn overlap(s: u64, d: u64, a: u64, b: u64) -> u64 {
    (s + d).min(b).saturating_sub(s.max(a))
}

fn to_json(events: &[Event], threads: &BTreeMap<u32, String>) -> Value {
    let name_of = |tid: u32| threads.get(&tid).cloned().unwrap_or_else(|| format!("worker-{tid}"));
    let main = threads.iter().find(|(_, n)| n.as_str() == "main").map(|(t, _)| *t);
    let first = events.iter().map(|e| e.ts).min().unwrap_or(0);
    let last = events.iter().map(|e| if let Kind::Span(d) = e.kind { e.ts + d } else { e.ts }).max().unwrap_or(0);

    let mut frames: Vec<(u64, u64)> = events
        .iter()
        .filter_map(|e| match e.kind {
            Kind::Span(d) if Some(e.tid) == main && e.name == FRAME => Some((e.ts, d)),
            _ => None,
        })
        .collect();
    frames.sort_unstable();
    let mut pending: Vec<(u64, f64)> = events
        .iter()
        .filter_map(|e| match e.kind {
            Kind::Counter(v) if e.name == CANVAS_PENDING => Some((e.ts, v)),
            _ => None,
        })
        .collect();
    pending.sort_by_key(|p| p.0);
    let mut actions: Vec<(u64, String)> = events.iter().filter(|e| matches!(e.kind, Kind::Action)).map(|e| (e.ts, e.name.to_string())).collect();
    actions.sort_by_key(|a| a.0);
    if actions.first().is_none_or(|a| a.0 > first) {
        actions.insert(0, (first, "(window start)".into()));
    }

    let mut summary = Vec::new();
    for (i, (at, name)) in actions.iter().enumerate() {
        let end = actions.get(i + 1).map_or(last, |a| a.0);
        let window: Vec<(u64, u64)> = frames.iter().copied().filter(|(s, _)| *s >= *at && *s < end).collect();
        let first_frame = window.first().map(|(s, d)| s + d);
        // Settled: the frame that first reported nothing pending after the last pending one.
        let inside: Vec<(u64, f64)> = pending.iter().copied().filter(|(t, _)| *t >= *at && *t < end).collect();
        let settled = match inside.iter().rposition(|(_, v)| *v > 0.0) {
            None => first_frame,
            Some(k) => inside.get(k + 1).map(|(t, _)| *t),
        };
        let stop = settled.unwrap_or(end);
        let shown: Vec<(u64, u64)> = window.iter().copied().filter(|(s, _)| *s < stop.max(*at + 1)).collect();
        let frame_ms: Vec<f64> = shown.iter().map(|(_, d)| ms(*d)).collect();
        let interval_ms: Vec<f64> = shown.windows(2).map(|w| ms(w[1].0 - w[0].0)).collect();
        let mut main_ms = BTreeMap::new();
        let mut background_ms = BTreeMap::new();
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        for e in events {
            match e.kind {
                Kind::Span(d) if e.name != FRAME => {
                    let us = overlap(e.ts, d, *at, stop);
                    if us == 0 {
                        continue;
                    }
                    if Some(e.tid) == main {
                        *main_ms.entry(e.name.to_string()).or_insert(0) += us;
                    } else {
                        *background_ms.entry(format!("{} / {}", name_of(e.tid), e.name)).or_insert(0) += us;
                    }
                }
                Kind::Instant if e.ts >= *at && e.ts < stop => *counts.entry(e.name.to_string()).or_insert(0) += 1,
                _ => {}
            }
        }
        let frames_total: u64 = shown.iter().map(|(_, d)| *d).sum();
        summary.push(json!({
            "t_s": ms(at - first) / 1000.0,
            "action": name,
            "first_frame_ms": first_frame.map(|f| ms(f - at)),
            "settled_ms": settled.map(|s| ms(s - at)),
            "unsettled": settled.is_none(),
            "frames": shown.len(),
            "frame_ms": stats(frame_ms),
            "frame_interval_ms": stats(interval_ms),
            "main_busy_in_frames_ms": ms(frames_total),
            "main_ms": top(main_ms, 16),
            "background_ms": top(background_ms, 16),
            "events": counts,
        }));
    }

    let mut trace = Vec::with_capacity(events.len() + threads.len());
    for (tid, name) in threads {
        trace.push(json!({ "ph": "M", "name": "thread_name", "pid": 1, "tid": tid, "args": { "name": name } }));
    }
    for e in events {
        let mut v = match e.kind {
            Kind::Span(d) => json!({ "ph": "X", "dur": d }),
            Kind::Action => json!({ "ph": "i", "s": "g", "cat": "action" }),
            Kind::Instant => json!({ "ph": "i", "s": "t" }),
            Kind::Counter(c) => json!({ "ph": "C", "args": { "value": c } }),
        };
        if let Value::Object(m) = &mut v {
            m.insert("name".into(), json!(e.name));
            m.insert("pid".into(), json!(1));
            m.insert("tid".into(), json!(e.tid));
            m.insert("ts".into(), json!(e.ts));
            if let Some(a) = &e.arg {
                m.insert("args".into(), json!({ "detail": a }));
            }
        }
        trace.push(v);
    }
    json!({
        "displayTimeUnit": "ms",
        "photocraft": { "version": 1, "window_s": ms(last - first) / 1000.0, "actions": summary },
        "traceEvents": trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(kind: Kind, name: &'static str, tid: u32, ts: u64) -> Event {
        Event { kind, name: Cow::Borrowed(name), tid, ts, arg: None }
    }

    #[test]
    fn summary_measures_first_frame_settle_and_breakdown() {
        let threads = BTreeMap::from([(1, "main".to_string()), (2, "photocraft-effects-0".to_string())]);
        let events = vec![
            ev(Kind::Action, "zoom 25%", 1, 1_000),
            ev(Kind::Span(10_000), FRAME, 1, 2_000),
            ev(Kind::Span(6_000), "sync_fx", 1, 3_000),
            ev(Kind::Counter(1.0), CANVAS_PENDING, 1, 12_000),
            ev(Kind::Span(40_000), "fx_inputs", 2, 5_000),
            ev(Kind::Instant, "fx_deferred", 1, 4_000),
            ev(Kind::Span(5_000), FRAME, 1, 30_000),
            ev(Kind::Counter(1.0), CANVAS_PENDING, 1, 35_000),
            ev(Kind::Span(5_000), FRAME, 1, 50_000),
            ev(Kind::Counter(0.0), CANVAS_PENDING, 1, 55_000),
            ev(Kind::Span(5_000), FRAME, 1, 70_000),
            ev(Kind::Counter(0.0), CANVAS_PENDING, 1, 75_000),
        ];
        let doc = to_json(&events, &threads);
        let a = &doc["photocraft"]["actions"][0];
        assert_eq!(a["action"], "zoom 25%");
        assert_eq!(a["first_frame_ms"], 11.0);
        assert_eq!(a["settled_ms"], 54.0);
        assert_eq!(a["frames"], 3);
        assert_eq!(a["main_ms"]["sync_fx"], 6.0);
        assert_eq!(a["background_ms"]["photocraft-effects-0 / fx_inputs"], 40.0);
        assert_eq!(a["events"]["fx_deferred"], 1);
        assert!(doc["traceEvents"].as_array().is_some_and(|t| t.len() == events.len() + threads.len()));
    }

    #[test]
    fn utc_stamp_formats_dates() {
        assert_eq!(utc_stamp(0), "19700101-000000");
        assert_eq!(utc_stamp(1_791_627_157), "20261010-101237");
    }
}
