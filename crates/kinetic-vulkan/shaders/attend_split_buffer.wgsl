// (maximum, sum), then the weighted values, for each token, head and subgroup.
@group(0) @binding(3) var<storage, read_write> split_kv: array<vec2<f32>>;

fn split_at(t: u32, head: u32, slot: u32) -> u32 {
    return ((t * shape.n_head + head) * SUBGROUPS + slot) * (HD / 2u + 1u);
}
