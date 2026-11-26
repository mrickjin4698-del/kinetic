// Two consecutive prompt queries of one head share their initial KV loads.
// They keep separate online softmax sums. As in the single query pass,
// each subgroup visits every fourth position in increasing order.
const TOKEN_ROWS: u32 = 2u;
const_assert HEADS == 1u;

@compute @workgroup_size(THREADS)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(subgroup_id) slot: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let t0 = group.y * TOKEN_ROWS;
    let count = batch[0];
    if t0 >= count {
        return;
    }
    let t1 = min(t0 + 1u, count - 1u);
    let head = group.x;
    let qb0 = t0 * shape.stride + head * shape.q_head_len;
    let qb1 = t1 * shape.stride + head * shape.q_head_len;
    let kv_head = head / (shape.n_head / shape.n_kv);
    let row_words = shape.n_kv * HD / 2u;
    let first = kv_head * HD / 2u + lane;

    var q0: array<vec2<f32>, PAIRS>;
    var q1: array<vec2<f32>, PAIRS>;
    var a0: array<vec2<f32>, PAIRS>;
    var a1: array<vec2<f32>, PAIRS>;
    for (var k = 0u; k < PAIRS; k++) {
        let e0 = qb0 + 2u * (lane + 32u * k);
        let e1 = qb1 + 2u * (lane + 32u * k);
        q0[k] = vec2<f32>(qkv[e0], qkv[e0 + 1u]) * shape.scale;
        q1[k] = vec2<f32>(qkv[e1], qkv[e1 + 1u]) * shape.scale;
    }
    var m0 = LOWEST;
    var m1 = LOWEST;
    var l0 = 0.0;
    var l1 = 0.0;

    // The full cache holds this batch's entire prefix, without wrapping.
    let len = batch[1] + t0 + 1u;
    var p = slot;
    for (; p < len; p += SUBGROUPS) {
        let row = p * row_words + first;
        var qk0 = 0.0;
        var qk1 = 0.0;
        var values: array<u32, PAIRS>;
        for (var k = 0u; k < PAIRS; k++) {
            let pair = load_for_score(row + 32u * k);
            let key = unpack2x16float(pair.x);
            qk0 += dot(q0[k], key);
            qk1 += dot(q1[k], key);
            values[k] = pair.y;
        }
        let s0 = subgroupAdd(qk0);
        let s1 = subgroupAdd(qk1);
        let next0 = max(m0, s0);
        let next1 = max(m1, s1);
        let c0 = exp(m0 - next0);
        let e0 = exp(s0 - next0);
        let c1 = exp(m1 - next1);
        let e1 = exp(s1 - next1);
        l0 = l0 * c0 + e0;
        l1 = l1 * c1 + e1;
        for (var k = 0u; k < PAIRS; k++) {
            let v = unpack2x16float(values[k]);
            a0[k] = a0[k] * c0 + e0 * v;
            a1[k] = a1[k] * c1 + e1 * v;
        }
        m0 = next0;
        m1 = next1;
    }

    // Only the second query sees this pair's extra position.
    if t1 != t0 && p == len {
        let row = p * row_words + first;
        var qk = 0.0;
        var values: array<u32, PAIRS>;
        for (var k = 0u; k < PAIRS; k++) {
            let pair = load_for_score(row + 32u * k);
            qk += dot(q1[k], unpack2x16float(pair.x));
            values[k] = pair.y;
        }
        let s = subgroupAdd(qk);
        let next = max(m1, s);
        let c = exp(m1 - next);
        let e = exp(s - next);
        l1 = l1 * c + e;
        for (var k = 0u; k < PAIRS; k++) {
            a1[k] = a1[k] * c + e * unpack2x16float(values[k]);
        }
        m1 = next;
    }

    save_partial(0u, slot, lane, m0, l0, a0);
    if t1 != t0 {
        save_partial(SUBGROUPS, slot, lane, m1, l1, a1);
    }
    workgroupBarrier();
    for (var t = t0; t <= t1; t++) {
        let part = (t - t0) * SUBGROUPS;
        write_output(t, head, t * shape.stride + head * shape.q_head_len, part, tid, lane);
    }
}
