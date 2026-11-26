// Prepare one pair of query and key normalization factors per token and
// key head. The raw channels remain available to ordinary passes and replay.
struct Shape {
    n_k_heads: u32,
    journal_slots: u32,
    journal_stride: u32,
    eps: f32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read_write> journal: array<vec4<f32>>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(subgroup_id) slot: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let first = batch[4];
    let count = batch[1] + batch[0] - first;
    let head = group.x * 4u + slot;
    if group.y >= count || head >= shape.n_k_heads {
        return;
    }
    let pos = first + group.y;
    let stride = shape.journal_stride / 4u;
    let base = pos % shape.journal_slots * stride;
    let query = base + head * (DK / 4u) + lane;
    let key = query + shape.n_k_heads * (DK / 4u);
    let qs = delta_norm_scale(journal[query], shape.eps);
    let ks = delta_norm_scale(journal[key], shape.eps);
    if lane == 0u {
        journal[base + stride - shape.n_k_heads + head] = vec4<f32>(qs, ks, 0.0, 0.0);
    }
}
