// Each element packs two f16 keys in x and the corresponding values in y.
@group(0) @binding(2) var<storage, read> kv: array<vec2<u32>>;

fn load_for_score(at: u32) -> vec2<u32> {
    return kv[at];
}

fn load_value(_at: u32, cached: u32) -> u32 {
    return cached;
}
