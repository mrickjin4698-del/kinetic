// Q8_0 on the matrix units: a block is one step. The device copy is
// planar, all quants then all f16 scales.

const SUB: u32 = 1u;

// Two bytes at a time, flip the sign bit to get q + 128 and put it
// in a half's mantissa at 1024, where the spacing is exactly one.
fn half_quants(word: u32) -> vec4<f16> {
    let lo = (word & 0x00ff00ffu) ^ 0x64806480u;
    let hi = ((word >> 8u) & 0x00ff00ffu) ^ 0x64806480u;
    let even = vec2<f16>(unpack2x16float(lo));
    let odd = vec2<f16>(unpack2x16float(hi));
    return vec4<f16>(even.x, odd.x, even.y, odd.y) - vec4<f16>(1152.0h);
}

// Weights 16h .. 16h + 16 of block `blk` to wa[at..].
fn stage(total: u32, blk: u32, s: u32, h: u32, at: u32) {
    let q = w[blk * 2u + h];
    let d = f16(f16_at(total * 2u, blk));
    for (var j = 0u; j < 4u; j++) {
        wa[at + j] = half_quants(q[j]) * d;
    }
}
