// The recurrence of a gated delta-rule layer. Each subgroup owns one or two
// rows of one value head's state, with four values per lane per row, and
// applies the delta rule
//     S = decay * S;  S += b * (v - S k) k^T
// for every position in turn, then reads out o = S q for the pass's tokens.
// Value head h reads key head h % n_k_heads.
//
// Positions run from batch word 4: before the pass's tokens come the tokens
// run since a checkpoint, replayed from their journal slots, so the state
// itself changes only when batch word 5 says to keep it. With `replay`, no
// token of the pass runs: the journal up to the pass's first position is
// applied and kept, which is how a rollback keeps the tokens it keeps.

struct Shape {
    channels: u32,
    n_k_heads: u32,
    n_v_heads: u32,
    head_v_dim: u32,
    journal_slots: u32,
    eps: f32,
    replay: u32,
    journal_stride: u32,
    prepared_norms: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read> journal: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> state: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

fn update(s: vec4<f32>, k: vec4<f32>, v: f32, decay: f32, b: f32) -> vec4<f32> {
    let delta = (v - decay * subgroupAdd(dot(s, k))) * b;
    return s * decay + k * delta;
}
