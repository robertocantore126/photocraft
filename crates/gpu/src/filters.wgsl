struct Params {
    size: vec4<u32>, // width, height, axis, radius
    values: vec4<f32>, // amount, threshold, motion dx, motion dy
    mode: vec4<u32>, // alpha, initial pass, final kind, motion steps
    origin: vec4<i32>,
};
@group(0) @binding(0) var<storage, read> source: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> destination: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> original: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn load(p: vec2<i32>) -> vec4<f32> {
    let q = clamp(p, vec2(0), vec2<i32>(params.size.xy) - vec2(1));
    var v = source[u32(q.y) * params.size.x + u32(q.x)];
    if (params.mode.x != 0u && params.mode.y != 0u) { v = vec4(v.rgb * v.a, v.a); }
    return v;
}

@compute @workgroup_size(8, 8)
fn convolution(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= params.size.xy)) { return; }
    var result = vec4(0.0);
    let r = i32(params.size.w);
    for (var i = -r; i <= r; i++) {
        let offset = select(vec2(0, i), vec2(i, 0), params.size.z == 0u);
        result += load(vec2<i32>(id.xy) + offset) * weights[u32(i + r)];
    }
    destination[id.y * params.size.x + id.x] = result;
}

// One invocation per row/column, matching the CPU running-sum box sequence.
// Work scales with line length plus initialization radius, rather than radius per pixel.
@compute @workgroup_size(64)
fn box_line(@builtin(global_invocation_id) id: vec3<u32>) {
    let horizontal = params.size.z == 0u;
    let count = select(params.size.x, params.size.y, horizontal);
    let length = select(params.size.y, params.size.x, horizontal);
    if (id.x >= count) { return; }
    let r = i32(params.size.w);
    var sum = vec4(0.0);
    for (var i = -r; i <= r; i++) {
        let p = select(vec2(i32(id.x), i), vec2(i, i32(id.x)), horizontal);
        sum += load(p);
    }
    let norm = 1.0 / f32(2 * r + 1);
    for (var x = 0; x < i32(length); x++) {
        let p = select(vec2(i32(id.x), x), vec2(x, i32(id.x)), horizontal);
        destination[u32(p.y) * params.size.x + u32(p.x)] = sum * norm;
        let add = select(vec2(0, r + 1), vec2(r + 1, 0), horizontal);
        let sub = select(vec2(0, r), vec2(r, 0), horizontal);
        sum += load(p + add) - load(p - sub);
    }
}

fn sample_zero(p: vec2<i32>) -> vec4<f32> {
    if (any(p < vec2(0)) || any(p >= vec2<i32>(params.size.xy))) { return vec4(0.0); }
    var v = source[u32(p.y) * params.size.x + u32(p.x)];
    if (params.mode.x != 0u) { v = vec4(v.rgb * v.a, v.a); }
    return v;
}

@compute @workgroup_size(8, 8)
fn motion(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= params.size.xy)) { return; }
    if (params.mode.w == 0u) { destination[id.y * params.size.x + id.x] = load(vec2<i32>(id.xy)); return; }
    var sum = vec4(0.0);
    for (var i = 0u; i <= params.mode.w; i++) {
        let t = f32(i) / f32(params.mode.w) - 0.5;
        let p = (vec2<f32>(id.xy) + vec2<f32>(params.origin.xy) + vec2(0.5)) + params.values.zw * t - vec2(0.5);
        let absolute = vec2<i32>(floor(p));
        let q = absolute - params.origin.xy;
        let f = p - vec2<f32>(absolute);
        // Same bilinear summation order as algo::Image::sample.
        var v = sample_zero(q) * ((1.0 - f.x) * (1.0 - f.y));
        v += sample_zero(q + vec2(1, 0)) * (f.x * (1.0 - f.y));
        v += sample_zero(q + vec2(0, 1)) * ((1.0 - f.x) * f.y);
        v += sample_zero(q + vec2(1, 1)) * (f.x * f.y);
        sum += v;
    }
    destination[id.y * params.size.x + id.x] = sum / f32(params.mode.w + 1u);
}

@compute @workgroup_size(8, 8)
fn finish(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= params.size.xy)) { return; }
    let i = id.y * params.size.x + id.x;
    var blurred = source[i];
    if (params.mode.x != 0u) {
        blurred = vec4(select(vec3(0.0), blurred.rgb / max(blurred.a, 1e-30), blurred.a > 1e-7), blurred.a);
    }
    if (params.mode.z == 0u) { destination[i] = blurred; return; }
    let old = original[i];
    if (params.mode.x != 0u && old.a <= 0.0) { destination[i] = old; return; }
    let delta = old - blurred;
    var result = vec4(0.5) + delta;
    if (params.mode.z == 1u) {
        result = select(old, old + params.values.x * delta, abs(delta) + vec4(1e-6) >= vec4(params.values.y));
    }
    if (params.mode.x != 0u) { result.a = old.a; }
    destination[i] = result;
}
