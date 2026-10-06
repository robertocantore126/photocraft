//! Kernel histogram of a document's GPU plan, and how many plain Content+Blend pairs run in chains
//! (each chain could be one fused pass).
use photocraft_gpu::Kernel;
use std::collections::BTreeMap;

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("usage: plan_stats FILE")?;
    if let Ok(dir) = std::env::var("PHOTOCRAFT_PROBE_SCRATCH") {
        photocraft_raster::spill::configure(Some(std::path::Path::new(&dir)), 2048 << 20)?;
    }
    let data = std::fs::read(&path).map_err(|e| e.to_string())?;
    let doc = photocraft_io::import_owned(&path, data).map_err(|e| e.to_string())?.document;
    for round in 0..3 {
        let t = std::time::Instant::now();
        let p = photocraft_gpu::plan(&doc).map_err(|e| e.to_string())?;
        println!("plan round {round}: {:.2} ms, {} passes", t.elapsed().as_secs_f64() * 1000.0, p.passes.len());
    }
    let plan = photocraft_gpu::plan(&doc).map_err(|e| e.to_string())?;
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for p in &plan.passes {
        *hist.entry(format!("{:?}", p.kernel)).or_default() += 1;
    }
    println!("passes {} slots {} fx layers {}", plan.passes.len(), plan.slots, plan.fx.len());
    for (k, n) in &hist {
        println!("  {k:12} {n}");
    }
    // Plain pairs: Content then Blend reading it as B; chained when the next pair's backdrop is
    // this pair's output.
    let (mut pairs, mut chains, mut longest, mut run, mut last_dst) = (0, 0, 0, 0, None);
    let mut i = 0;
    let mut lens: BTreeMap<usize, usize> = BTreeMap::new();
    while i + 1 < plan.passes.len() {
        let (c, b) = (&plan.passes[i], &plan.passes[i + 1]);
        if c.kernel == Kernel::Content && b.kernel == Kernel::Blend && b.b == Some(c.dst) && c.clip.is_none() && b.clip.is_none() {
            pairs += 1;
            if run > 0 && b.a == last_dst {
                run += 1;
            } else {
                if run > 0 {
                    *lens.entry(run).or_default() += 1;
                }
                chains += 1;
                run = 1;
            }
            longest = longest.max(run);
            last_dst = Some(b.dst);
            i += 2;
        } else {
            if run > 0 {
                *lens.entry(run).or_default() += 1;
            }
            run = 0;
            last_dst = None;
            i += 1;
        }
    }
    if run > 0 {
        *lens.entry(run).or_default() += 1;
    }
    println!("plain pairs {pairs} in {chains} chains, longest {longest}");
    println!("chain lengths: {lens:?}");
    // Where each named layer's pixels are first read: passes before it don't depend on it.
    for name in std::env::args().skip(2) {
        let ids: Vec<_> = doc.walk().into_iter().filter(|(_, _, l)| l.name == name).map(|(_, _, l)| l.id).collect();
        for id in ids {
            let first = plan.passes.iter().position(|p| p.tex.as_ref().is_some_and(|t| t.layer == id) || p.mask.as_ref().is_some_and(|m| m.layer == id));
            println!("{name} ({id:?}): first pass {first:?} of {}", plan.passes.len());
        }
    }
    Ok(())
}
