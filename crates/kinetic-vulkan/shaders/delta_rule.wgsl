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

const PAIRED: bool = ROWS_PER_SUBGROUP == 2u;

fn update(s: vec4<f32>, k: vec4<f32>, v: f32, decay: f32, b: f32) -> vec4<f32> {
    let delta = (v - decay * subgroupAdd(dot(s, k))) * b;
    return s * decay + k * delta;
}

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(subgroup_id) slot: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let dv = shape.head_v_dim;
    let row = (group.x * 4u + slot) * ROWS_PER_SUBGROUP;
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
    let factors_at = (shape.channels + 2u * vh) / 4u;
    let high_pair = vh % 2u != 0u;
    let jstride = shape.journal_stride / 4u;
    let pos0 = batch[1];
    var end = pos0 + batch[0];
    if shape.replay != 0u {
        end = pos0;
    }
    let q_scale = 1.0 / sqrt(f32(DK));

    let at = row * (DK / 4u) + lane;
    var s = state[at];
    var s1: vec4<f32>;
    if PAIRED {
        s1 = state[at + DK / 4u];
    }
    for (var p = batch[4]; p < end; p++) {
        var journal_slot = p & (shape.journal_slots - 1u);
        if !MASK_SLOTS {
            journal_slot = p % shape.journal_slots;
        }
        let base = journal_slot * jstride;
        let raw_k = journal[base + key_at];
        var scales: vec2<f32>;
        if shape.prepared_norms != 0u {
            scales = journal[base + jstride - shape.n_k_heads + kh].xy;
        } else {
            scales.y = delta_norm_scale(raw_k, shape.eps);
        }
        let k = raw_k * scales.y;
        let values = journal[base + value_at / 4u];
        let index = value_at % 4u;
        let word = journal[base + factors_at];
        let factors = select(word.xy, word.zw, high_pair);
        let decay = factors.x;
        let b = factors.y;
        s = update(s, k, values[index], decay, b);
        if PAIRED {
            // The value head width is a multiple of 32. Both rows share
            // the head and their values occupy the same journal vector.
            s1 = update(s1, k, values[index + 1u], decay, b);
        }
        if p >= pos0 {
            let raw_q = journal[base + query_at];
            if shape.prepared_norms == 0u {
                scales.x = delta_norm_scale(raw_q, shape.eps);
            }
            let q = raw_q * scales.x * q_scale;
            let o = subgroupAdd(dot(s, q));
            var o1: f32;
            if PAIRED {
                o1 = subgroupAdd(dot(s1, q));
            }
            if lane == 0u {
                let out_at = (p - pos0) * value_dim + row;
                out[out_at] = o;
                if PAIRED {
                    out[out_at + 1u] = o1;
                }
            }
        }
    }
    if batch[5] != 0u {
        state[at] = s;
        if PAIRED {
            state[at + DK / 4u] = s1;
        }
    }
}
