// Same seed threshold and scalar field encoding as compose::effects.
// JFA+1 nearest-seed selection is approximate; strokes never enter this shader.
struct Params {
    width: u32,
    height: u32,
    stride: u32,
    step: u32,
    kind: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}
@group(0) @binding(0) var<storage, read> source: array<vec2<i32>>;
@group(0) @binding(1) var<storage, read_write> dest: array<vec2<i32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var shape: texture_2d<f32>;
@group(0) @binding(4) var<uniform> p: Params;

fn alpha(at: vec2<i32>) -> f32 {
    let a = textureLoad(shape, at, 0).r;
    return select(a, 1.0 - a, p.kind == 2u);
}
fn is_seed(at: vec2<i32>) -> bool {
    let a = alpha(at);
    return select(a > 0.5 / 255.0, a <= 0.5 / 255.0, p.kind == 1u);
}
fn distance_squared(at: vec2<i32>, candidate: vec2<i32>) -> i32 {
    let d = at - candidate;
    return d.x * d.x + d.y * d.y;
}
@compute @workgroup_size(8, 8)
fn seed(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.width || id.y >= p.height { return; }
    let at = vec2<i32>(id.xy);
    dest[id.y * p.width + id.x] = select(vec2<i32>(-1), at, is_seed(at));
}
@compute @workgroup_size(8, 8)
fn jump(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.width || id.y >= p.height { return; }
    let at = vec2<i32>(id.xy);
    var best = vec2<i32>(-1);
    var best_distance = 2147483647;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let neighbour = at + vec2<i32>(x, y) * i32(p.step);
            if any(neighbour < vec2<i32>(0)) || any(neighbour >= vec2<i32>(i32(p.width), i32(p.height))) { continue; }
            let candidate = source[u32(neighbour.y) * p.width + u32(neighbour.x)];
            if candidate.x < 0 { continue; }
            let d = distance_squared(at, candidate);
            // EDT ties choose the earlier column, then the earlier row.
            let earlier = candidate.x < best.x || (candidate.x == best.x && candidate.y < best.y);
            if d < best_distance || (d == best_distance && earlier) {
                best = candidate;
                best_distance = d;
            }
        }
    }
    dest[id.y * p.width + id.x] = best;
}
@compute @workgroup_size(8, 8)
fn finish(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.width || id.y >= p.height { return; }
    let at = vec2<i32>(id.xy);
    let nearest = source[id.y * p.width + id.x];
    var d = -0.5;
    if !is_seed(at) {
        // EDT uses sqrt(1e20) when there are no seeds. Its default nearest index is zero.
        var distance = 1e10;
        var coverage = alpha(vec2<i32>(0));
        if nearest.x >= 0 {
            distance = sqrt(f32(distance_squared(at, nearest)));
            coverage = alpha(nearest);
        }
        d = distance - 0.5;
        if p.kind != 1u { d += 1.0 - min(coverage, 1.0); }
    }
    output[id.y * p.stride + id.x] = d;
}
