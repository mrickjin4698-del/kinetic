// Q6_K on the matrix units, from the planar device copy (see gemv_q6_k):
// step s of a block is elements 32s .. 32s + 32, group g = s % 4 of half
// n = s / 4. Element 16h + i of the step takes its low four bits from ql
// byte 64n + 32(g & 1) + 16h + i (high nibble for g >= 2), its top two from
// bits 2g of qh byte 32n + 16h + i, and int8 scale 2s + h; it is worth
// d * scale * (q - 32).

const SUB: u32 = 8u;

// Signed byte `i` of a word.
fn sbyte(v: vec4<u32>, i: u32) -> f32 {
    return f32(extractBits(bitcast<i32>(v[i / 4u]), 8u * (i % 4u), 8u));
}

// Weights 16h .. 16h + 16 of step `s` of block `blk` to wa[at..].
fn stage(total: u32, blk: u32, s: u32, h: u32, at: u32) {
    let n = s / 4u;
    let g = s % 4u;
    let lq = w[blk * 8u + 4u * n + 2u * (g & 1u) + h];
    let hq = w[total * 8u + blk * 4u + 2u * n + h];
    let scale = f16_at(total * 13u, blk) * sbyte(w[total * 12u + blk], 2u * s + h);
    let lshift = 4u * (g / 2u);
    let hshift = 2u * g;
    for (var j = 0u; j < 4u; j++) {
        let lo = unpack4xU8(lq[j] >> lshift) & vec4<u32>(15u);
        let hi = (unpack4xU8(hq[j] >> hshift) & vec4<u32>(3u)) << vec4<u32>(4u);
        wa[at + j] = vec4<f16>(scale * (vec4<f32>(lo | hi) - vec4<f32>(32.0)));
    }
}
