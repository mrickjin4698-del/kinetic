@group(0) @binding(2) var<storage, read> kc: array<u32>;
@group(0) @binding(3) var<storage, read> vc: array<u32>;

fn load_for_score(at: u32) -> vec2<u32> {
    return vec2<u32>(kc[at], 0u);
}

fn load_value(at: u32, _cached: u32) -> u32 {
    return vc[at];
}
