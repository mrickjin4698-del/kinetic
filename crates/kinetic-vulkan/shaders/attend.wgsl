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
    let p = scan(t, head, slot, lane, HEADS == 2u);
    let qb = t * shape.stride + head * shape.q_head_len;
    save_partial(part, slot, lane, p.top, p.total, p.values);
    workgroupBarrier();
    write_output(t, head, qb, part, tid, lane);
}
