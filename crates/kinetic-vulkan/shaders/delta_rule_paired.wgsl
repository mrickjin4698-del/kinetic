const LANES: u32 = DK / 4u;
const SUBGROUPS: u32 = 4u;
const TILE_STEPS: u32 = 8u;

var<workgroup> pass_words: vec4<u32>;
var<workgroup> queries: array<vec4<f32>, TILE_STEPS * LANES>;
var<workgroup> keys: array<vec4<f32>, TILE_STEPS * LANES>;
var<workgroup> values: array<vec4<f32>, TILE_STEPS * 2u>;
var<workgroup> coefficients: array<vec2<f32>, TILE_STEPS>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(subgroup_id) slot: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    // All eight state rows belong to one value head. Its query and key
    // vectors and coefficients can be staged once for the whole group.
    let row0 = group.x * (SUBGROUPS * 2u);
    let row = row0 + slot * 2u;
    let dv = shape.head_v_dim;
    let vh = row0 / dv;
    let kh = vh % shape.n_k_heads;
    let key_dim = shape.n_k_heads * DK;
    let value_dim = shape.n_v_heads * dv;
    let query_at = kh * LANES + lane;
    let key_at = key_dim / 4u + query_at;
    let value_at = (2u * key_dim + row0) / 4u;
    let factors_at = (shape.channels + 2u * vh) / 4u;
    let high_pair = vh % 2u != 0u;
    let jstride = shape.journal_stride / 4u;
    let q_scale = 1.0 / sqrt(f32(DK));

    // All subgroups must visit the same tiles, including any checkpoint prefix.
    if tid == 0u {
        pass_words = vec4<u32>(batch[4], batch[1], batch[1] + batch[0], batch[5]);
    }
    let span = workgroupUniformLoad(&pass_words);
    let pos0 = span.y;
    let end = span.z;

    let at = row * LANES + lane;
    var s = state[at];
    var s1 = state[at + LANES];
    for (var start = span.x; start < end;) {
        let count = min(TILE_STEPS, end - start);
        for (var step = slot; step < count; step += SUBGROUPS) {
            let base = ((start + step) & (shape.journal_slots - 1u)) * jstride;
            let scales = journal[base + jstride - shape.n_k_heads + kh].xy;
            let shared_at = step * LANES + lane;
            queries[shared_at] = journal[base + query_at] * scales.x * q_scale;
            keys[shared_at] = journal[base + key_at] * scales.y;
            if lane == 0u {
                values[step * 2u] = journal[base + value_at];
                values[step * 2u + 1u] = journal[base + value_at + 1u];
                let word = journal[base + factors_at];
                coefficients[step] = select(word.xy, word.zw, high_pair);
            }
        }
        workgroupBarrier();

        let index = (slot % 2u) * 2u;
        for (var step = 0u; step < count; step++) {
            let shared_at = step * LANES + lane;
            let q = queries[shared_at];
            let k = keys[shared_at];
            let v = values[step * 2u + slot / 2u];
            let factors = coefficients[step];
            s = update(s, k, v[index], factors.x, factors.y);
            s1 = update(s1, k, v[index + 1u], factors.x, factors.y);

            let p = start + step;
            if p >= pos0 {
                let o = subgroupAdd(dot(s, q));
                let o1 = subgroupAdd(dot(s1, q));
                if lane == 0u {
                    let out_at = (p - pos0) * value_dim + row;
                    out[out_at] = o;
                    out[out_at + 1u] = o1;
                }
            }
        }
        start += count;
        // Keep the tile available until every subgroup finishes reading it.
        workgroupBarrier();
    }

    if span.w != 0u {
        state[at] = s;
        state[at + LANES] = s1;
    }
}
