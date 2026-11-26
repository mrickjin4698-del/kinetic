// Attention of one token over its visible cached positions, written as a
// GEMV input. The query heads of a workgroup share one KV head and run
// on the same compute unit. Lane i holds pairs i + 32k of each head.
//
// Each query head gets four subgroups that take every fourth position.
// Each keeps a running maximum, sum and weighted sum of values (an
// online softmax), and the four combine at the end. A token's result
// does not depend on the rest of its batch.
//
// A token sees the last `window` positions up to its own; position p sits
// in cache slot p % slots. With gate 1 the output is multiplied by the
// sigmoid of the gate values after each query head; with gate 2 by that of
// the head's one gate value at `gate_at` in the token's row.

const TOKEN_ROWS: u32 = 1u;

fn next_offset(offset: u32, step: u32, wrap_at: u32) -> u32 {
    return select(offset + step, offset - wrap_at, offset >= wrap_at);
}

@compute @workgroup_size(THREADS * HEADS)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
    @builtin(subgroup_id) subgroup: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let t = group.y;
    if t >= batch[0] {
        return;
    }
    let in_group = subgroup / SUBGROUPS;
    let part = in_group * SUBGROUPS;
    let slot = subgroup % SUBGROUPS;
    let tid = select(local, slot * 32u + lane, HEADS > 1u);
    let head = group.x * HEADS + in_group;
    let len = batch[1] + t + 1u;
    let start = len - min(len, shape.window);
    let row_words = shape.n_kv * HD / 2u;
    let kv_head = head / (shape.n_head / shape.n_kv);
    let qb = t * shape.stride + head * shape.q_head_len;

    var q: array<vec2<f32>, PAIRS>;
    var acc: array<vec2<f32>, PAIRS>;
    for (var k = 0u; k < PAIRS; k++) {
        let e = qb + 2u * (lane + 32u * k);
        q[k] = vec2<f32>(qkv[e], qkv[e + 1u]) * shape.scale;
        acc[k] = vec2<f32>(0.0);
    }
    var m = LOWEST;
    var l = 0.0;
    let first = kv_head * HD / 2u + lane;
    let first_pos = start + slot;
    // Walk the ring in words, wrapping once if the stride crosses its end.
    let cache_step = SUBGROUPS % shape.slots;
    let word_step = cache_step * row_words;
    let wrap_at = (shape.slots - cache_step) * row_words;
    var cache_offset = (first_pos % shape.slots) * row_words;
    var p = first_pos;
    if HEADS == 2u {
        // Interleave two reads, then apply the two softmax updates in order.
        let pair_end = len - min(len, SUBGROUPS);
        for (; p < pair_end; p += 2u * SUBGROUPS) {
            let offset1 = next_offset(cache_offset, word_step, wrap_at);
            let row0 = cache_offset + first;
            let row1 = offset1 + first;
            var qk0 = 0.0;
            var qk1 = 0.0;
            var values: array<vec2<u32>, PAIRS>;
            for (var k = 0u; k < PAIRS; k++) {
                let pair = 32u * k;
                let a = load_for_score(row0 + pair);
                let b = load_for_score(row1 + pair);
                values[k] = vec2<u32>(a.y, b.y);
                qk0 += dot(q[k], unpack2x16float(a.x));
                qk1 += dot(q[k], unpack2x16float(b.x));
            }
            let s0 = subgroupAdd(qk0);
            let s1 = subgroupAdd(qk1);
            let next0 = max(m, s0);
            let next1 = max(next0, s1);
            let c0 = exp(m - next0);
            let e0 = exp(s0 - next0);
            let c1 = exp(next0 - next1);
            let e1 = exp(s1 - next1);
            l = l * c0 + e0;
            l = l * c1 + e1;
            for (var k = 0u; k < PAIRS; k++) {
                let pair = 32u * k;
                let cached = values[k];
                let v0 = unpack2x16float(load_value(row0 + pair, cached.x));
                let v1 = unpack2x16float(load_value(row1 + pair, cached.y));
                acc[k] = acc[k] * c0 + e0 * v0;
                acc[k] = acc[k] * c1 + e1 * v1;
            }
            m = next1;
            cache_offset = next_offset(offset1, word_step, wrap_at);
        }
    }
    for (; p < len; p += SUBGROUPS) {
        let row = cache_offset + first;
        var qk = 0.0;
        var values: array<u32, PAIRS>;
        for (var k = 0u; k < PAIRS; k++) {
            let pair = load_for_score(row + 32u * k);
            values[k] = pair.y;
            qk += dot(q[k], unpack2x16float(pair.x));
        }
        let s = subgroupAdd(qk);
        let next = max(m, s);
        let c = exp(m - next);
        let e = exp(s - next);
        l = l * c + e;
        for (var k = 0u; k < PAIRS; k++) {
            acc[k] = acc[k] * c + e * unpack2x16float(load_value(row + 32u * k, values[k]));
        }
        m = next;
        cache_offset = next_offset(cache_offset, word_step, wrap_at);
    }
    save_partial(part, slot, lane, m, l, acc);
    workgroupBarrier();
    write_output(t, head, qb, part, tid, lane);
}
