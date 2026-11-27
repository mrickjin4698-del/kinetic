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
    let factors_at = (shape.channels + 2u * vh) / 4u;
    let high_pair = vh % 2u != 0u;
    let jstride = shape.journal_stride / 4u;
    let pos0 = batch[1];
    var end = pos0 + batch[0];
    if shape.replay != 0u {
        end = pos0;
    }
    let prepared = shape.prepared_norms != 0u;
    let q_scale = 1.0 / sqrt(f32(DK));

    let at = row * (DK / 4u) + lane;
    var s = state[at];
    for (var p = batch[4]; p < end; p++) {
        var journal_slot = p & (shape.journal_slots - 1u);
        if !MASK_SLOTS {
            journal_slot = p % shape.journal_slots;
        }
        let base = journal_slot * jstride;
        let raw_k = journal[base + key_at];
        var scales: vec2<f32>;
        if prepared {
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
        if p >= pos0 {
            let raw_q = journal[base + query_at];
            if !prepared {
                scales.x = delta_norm_scale(raw_q, shape.eps);
            }
            let q = raw_q * scales.x * q_scale;
            let o = subgroupAdd(dot(s, q));
            if lane == 0u {
                let out_at = (p - pos0) * value_dim + row;
                out[out_at] = o;
            }
        }
    }
    if batch[5] != 0u {
        state[at] = s;
    }
}
