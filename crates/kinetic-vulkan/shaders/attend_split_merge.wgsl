const TOKEN_ROWS: u32 = 1u;

@compute @workgroup_size(THREADS)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
    @builtin(subgroup_id) slot: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let t = group.y;
    if t >= batch[0] {
        return;
    }
    let head = group.x;
    let at = split_at(t, head, slot);
    let stats = split_kv[at];
    var acc: array<vec2<f32>, PAIRS>;
    for (var k = 0u; k < PAIRS; k++) {
        acc[k] = split_kv[at + 1u + lane + 32u * k];
    }
    save_partial(0u, slot, lane, stats.x, stats.y, acc);
    workgroupBarrier();
    let qb = t * shape.stride + head * shape.q_head_len;
    write_output(t, head, qb, 0u, local, lane);
}
