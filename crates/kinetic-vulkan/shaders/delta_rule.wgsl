// The recurrence of a gated delta-rule layer. Each subgroup owns one row of
// one value head's state, `head_k_dim` = 128 values, four per lane, and
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
}

const DK: u32 = 128u;

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read> journal: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> state: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

fn journal_word(base: u32, word: u32) -> f32 {
    return journal[base + word / 4u][word % 4u];
}

fn l2_normed(x: vec4<f32>) -> vec4<f32> {
    let norm = sqrt(subgroupAdd(dot(x, x)));
    return x * (1.0 / max(norm, shape.eps));
}

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(subgroup_id) slot: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let dv = shape.head_v_dim;
    let row = group.x * 4u + slot;
    if row >= shape.n_v_heads * dv {
        return;
    }
    let vh = row / dv;
    let kh = vh % shape.n_k_heads;
    let key_dim = shape.n_k_heads * DK;
    let value_dim = shape.n_v_heads * dv;
    let query_at = kh * (DK / 4u) + lane;
    let key_at = key_dim / 4u + query_at;
    let value_at = 2u * key_dim + row;
    let decay_at = shape.channels + vh;
    let beta_at = decay_at + shape.n_v_heads;
    let jstride = shape.journal_stride / 4u;
    let pos0 = batch[1];
    var end = pos0 + batch[0];
    if shape.replay != 0u {
        end = pos0;
    }
    let q_scale = 1.0 / sqrt(f32(DK));

    let at = row * (DK / 4u) + lane;
    var s = state[at];
    for (var p = batch[4]; p < end; p++) {
        let base = p % shape.journal_slots * jstride;
        let k = l2_normed(journal[base + key_at]);
        let v = journal_word(base, value_at);
        let decay = journal_word(base, decay_at);
        let b = journal_word(base, beta_at);
        let delta = (v - decay * subgroupAdd(dot(s, k))) * b;
        s = s * decay + k * delta;
        if p >= pos0 {
            let q = l2_normed(journal[base + query_at]) * q_scale;
            let o = subgroupAdd(dot(s, q));
            if lane == 0u {
                out[(p - pos0) * value_dim + row] = o;
            }
        }
    }
    if batch[5] != 0u {
        state[at] = s;
    }
}
