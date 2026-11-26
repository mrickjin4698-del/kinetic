// Q4_K on the matrix units: step s of a block is sub-block s. Element l of
// sub-block s is nibble s % 2 of byte 32(s / 2) + l of the quants, worth
// d * scale * q - dmin * min.

const SUB: u32 = 8u;

// Weights 16h .. 16h + 16 of sub-block `s` of block `blk` to wa[at..].
fn stage(total: u32, blk: u32, s: u32, h: u32, at: u32) {
    let head = w[blk * 9u];
    let q = w[blk * 9u + 1u + 2u * (s / 2u) + h];
    let sm = k4_scale_min(s, head.y, head.z, head.w);
    let dm = unpack2x16float(head.x);
    let a = dm.x * sm.x;
    let b = dm.y * sm.y;
    let shift = 4u * (s % 2u);
    for (var j = 0u; j < 4u; j++) {
        let v = vec4<f32>(unpack4xU8(q[j] >> shift) & vec4<u32>(15u));
        wa[at + j] = vec4<f16>(a * v - vec4<f32>(b));
    }
}
