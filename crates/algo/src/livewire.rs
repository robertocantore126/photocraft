//! The Magnetic Lasso's live-wire (ported from Fotox): the cheapest 8-connected path between two
//! pixels of a window, the cost of a pixel being low on strong edges (Dijkstra on a
//! gradient-magnitude cost).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// The per-pixel cost of a luminance window: `1 − g` where `g` is the gradient magnitude
/// normalised to the window's strongest edge, with edges weaker than `contrast` (`0..=1`) ignored;
/// plus a small constant so the path stays short where there is no edge.
pub fn edge_cost(lum: &[f32], w: usize, h: usize, contrast: f32) -> Vec<f32> {
    let mut grad = vec![0.0f32; w * h];
    let mut max = 1e-6f32;
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            // Sobel.
            let gx = (lum[i - w + 1] + 2.0 * lum[i + 1] + lum[i + w + 1]) - (lum[i - w - 1] + 2.0 * lum[i - 1] + lum[i + w - 1]);
            let gy = (lum[i + w - 1] + 2.0 * lum[i + w] + lum[i + w + 1]) - (lum[i - w - 1] + 2.0 * lum[i - w] + lum[i - w + 1]);
            let g = (gx * gx + gy * gy).sqrt();
            grad[i] = g;
            max = max.max(g);
        }
    }
    let c = contrast.clamp(0.0, 0.95);
    grad.iter()
        .map(|g| {
            let n = ((g / max - c) / (1.0 - c)).clamp(0.0, 1.0);
            1.0 - n + 0.02
        })
        .collect()
}

#[derive(PartialEq)]
struct Node(f32, usize);
impl Eq for Node {}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The cheapest path from `start` to `end` (window pixels), both included, in order. Empty when
/// either is outside the window.
pub fn live_wire(cost: &[f32], w: usize, h: usize, start: (usize, usize), end: (usize, usize)) -> Vec<(usize, usize)> {
    if start.0 >= w || start.1 >= h || end.0 >= w || end.1 >= h || cost.len() < w * h {
        return Vec::new();
    }
    let s = start.1 * w + start.0;
    let e = end.1 * w + end.0;
    let mut dist = vec![f32::INFINITY; w * h];
    let mut prev = vec![usize::MAX; w * h];
    let mut heap = BinaryHeap::new();
    dist[s] = 0.0;
    heap.push(Node(0.0, s));
    while let Some(Node(d, i)) = heap.pop() {
        if i == e {
            break;
        }
        if d > dist[i] {
            continue;
        }
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                continue;
            }
            let j = ny as usize * w + nx as usize;
            let step = if dx != 0 && dy != 0 { std::f32::consts::SQRT_2 } else { 1.0 };
            let nd = d + cost[j] * step;
            if nd < dist[j] {
                dist[j] = nd;
                prev[j] = i;
                heap.push(Node(nd, j));
            }
        }
    }
    if prev[e] == usize::MAX && e != s {
        return vec![start, end];
    }
    let mut path = vec![end];
    let mut i = e;
    while i != s {
        i = prev[i];
        path.push((i % w, i / w));
    }
    path.reverse();
    path
}
