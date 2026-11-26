enable f16;

// Pack one token per workgroup. The f16 blocks of 32 inputs are laid out
// as [block][token][word of four f16s] for the matrix staging loops.
struct Shape {
    blocks: u32,
}

var<immediate> shape: Shape;
@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> xd: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> packed: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read_write> inv_scale: array<vec2<f32>>;

const THREADS: u32 = 128u;
var<workgroup> count: u32;
var<workgroup> peaks: array<f32, 4>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(subgroup_id) slot: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    if tid == 0u {
        count = batch[0];
    }
    let tokens = workgroupUniformLoad(&count);
    let row = group.x;
    if row >= tokens {
        return;
    }
    let steps = shape.blocks;
    var peak = 0.0;
    for (var b = tid; b < steps; b += THREADS) {
        peak = max(peak, xd[row * steps + b].x);
    }
    let maximum = subgroupMax(peak);
    if lane == 0u {
        peaks[slot] = maximum;
    }
    workgroupBarrier();
    let top = max(max(peaks[0], peaks[1]), max(peaks[2], peaks[3]));
    let sx = matrix_input_scale(top * 127.0);
    if tid == 0u {
        inv_scale[row] = vec2<f32>(1.0 / sx, 0.0);
    }
    for (var i = tid; i < steps * 8u; i += THREADS) {
        let at = (i / 8u * tokens + row) * 8u + i % 8u;
        packed[at] = vec4<f16>(x[row * steps * 8u + i] * sx);
    }
}
