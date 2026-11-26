struct Shape {
    n_head: u32,
    n_kv: u32,
    stride: u32,
    q_head_len: u32,
    scale: f32,
    gate: u32,
    gate_at: u32,
    window: u32,
    slots: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(4) var<storage, read_write> ox: array<f32>;
@group(0) @binding(5) var<storage, read_write> oq: array<u32>;
@group(0) @binding(6) var<storage, read_write> od: array<vec2<f32>>;

const HD: u32 = 64u * PAIRS;
const SUBGROUPS: u32 = 4u;
const THREADS: u32 = 32u * SUBGROUPS;
const LOWEST: f32 = -3.4e38;

var<workgroup> top: array<f32, TOKEN_ROWS * HEADS * SUBGROUPS>;
var<workgroup> total: array<f32, TOKEN_ROWS * HEADS * SUBGROUPS>;
var<workgroup> partial: array<f32, TOKEN_ROWS * HEADS * SUBGROUPS * HD>;

fn save_partial(part: u32, slot: u32, lane: u32, m: f32, l: f32, acc: array<vec2<f32>, PAIRS>) {
    if lane == 0u {
        top[part + slot] = m;
        total[part + slot] = l;
    }
    for (var k = 0u; k < PAIRS; k++) {
        let e = (part + slot) * HD + 2u * (lane + 32u * k);
        partial[e] = acc[k].x;
        partial[e + 1u] = acc[k].y;
    }
}

fn write_output(t: u32, head: u32, qb: u32, part: u32, tid: u32, lane: u32) {
    var best = top[part];
    for (var s = 1u; s < SUBGROUPS; s++) {
        best = max(best, top[part + s]);
    }
    var den = 0.0;
    for (var s = 0u; s < SUBGROUPS; s++) {
        den += total[part + s] * exp(top[part + s] - best);
    }
    let out = (t * shape.n_head + head) * HD;
    // HD is a multiple of 64, so subgroups run whole.
    for (var e = tid; e < HD; e += THREADS) {
        var o = 0.0;
        for (var s = 0u; s < SUBGROUPS; s++) {
            o += partial[(part + s) * HD + e] * exp(top[part + s] - best);
        }
        o /= den;
        if shape.gate == 1u {
            o /= 1.0 + exp(-qkv[qb + HD + e]);
        } else if shape.gate == 2u {
            o /= 1.0 + exp(-qkv[t * shape.stride + shape.gate_at + head]);
        }
        put_q8(out + e, o, lane);
    }
}
