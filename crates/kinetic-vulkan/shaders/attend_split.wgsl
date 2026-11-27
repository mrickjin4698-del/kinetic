const TOKEN_ROWS: u32 = 1u;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let t = group.y;
    if t >= batch[0] {
        return;
    }
    let head = group.x / SUBGROUPS;
    let slot = group.x % SUBGROUPS;
    let p = scan(t, head, slot, lane, true);
    let at = split_at(t, head, slot);
    if lane == 0u {
        split_kv[at] = vec2<f32>(p.top, p.total);
    }
    for (var k = 0u; k < PAIRS; k++) {
        split_kv[at + 1u + lane + 32u * k] = p.values[k];
    }
}
