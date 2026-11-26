// The front of a gated delta-rule layer for each token row of the batch:
// the causal depthwise convolution, with SiLU, of the projected queries,
// keys and values, and each value head's decay and step size. A row of
// `proj` holds the `channels` convolution inputs, then `value_dim` output
// gates, then one beta and one alpha per value head. The results go to the
// token's journal slot, position % journal_slots, which delta_rule.wgsl
// reads: the convolved channels, then (decay, step size) for each value head.
//
// Raw inputs are kept in a ring as in short_conv.wgsl, so neither a pass
// nor a rollback overwrites a position still read.

struct Shape {
    channels: u32,
    stride: u32,
    value_dim: u32,
    n_v_heads: u32,
    taps: u32,
    slots: u32,
    journal_slots: u32,
    journal_stride: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read> proj: array<f32>;
// dt_bias, then a = -exp(A_log), one per value head.
@group(0) @binding(3) var<storage, read> heads: array<f32>;
@group(0) @binding(4) var<storage, read_write> ring: array<f32>;
@group(0) @binding(5) var<storage, read_write> journal: array<f32>;

fn input(p: u32, pos0: u32, ch: u32) -> f32 {
    if p >= pos0 {
        return proj[(p - pos0) * shape.stride + ch];
    }
    return ring[p % shape.slots * shape.channels + ch];
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        return x;
    }
    return log(1.0 + exp(x));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let ch = group.x * 256u + tid;
    let t = group.y;
    if t >= batch[0] || ch >= shape.channels + shape.n_v_heads {
        return;
    }
    let pos0 = batch[1];
    let pos = pos0 + t;
    let slot = pos % shape.journal_slots * shape.journal_stride;
    if ch >= shape.channels {
        let h = ch - shape.channels;
        let at = t * shape.stride + shape.channels + shape.value_dim;
        let beta = proj[at + h];
        let alpha = proj[at + shape.n_v_heads + h];
        let decay = exp(softplus(alpha + heads[h]) * heads[shape.n_v_heads + h]);
        let factors = slot + shape.channels + 2u * h;
        journal[factors] = decay;
        journal[factors + 1u] = 1.0 / (1.0 + exp(-beta));
        return;
    }
    let x = proj[t * shape.stride + ch];
    let acc = convolve(t, pos0, ch, x);
    // The next pass reads only the last taps - 1 inputs of a committed pass.
    // While checkpointed, preserve all inputs so any accepted prefix works.
    if !COMMITTED_TAIL_ONLY || batch[5] == 0u || batch[0] - t < shape.taps {
        ring[pos % shape.slots * shape.channels + ch] = x;
    }
    journal[slot + ch] = acc / (1.0 + exp(-acc));
}
