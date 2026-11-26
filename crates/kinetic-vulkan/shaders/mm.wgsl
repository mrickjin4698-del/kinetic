enable f16;
enable wgpu_cooperative_matrix;

// Shared part of every kernel on the matrix units; build.rs appends one
// weight format to it, which defines `SUB`, the steps of 32 columns per
// block, and `stage`, which expands 16 weights of a step to f16.
//
// Y = W X for a pass of many tokens, on the matrix units of devices with
// cooperative 16x16x16 f16 products into f32. A workgroup of four
// subgroups takes 64 or 128 rows of W by 64 tokens. Each step stages 32
// columns of weights and inputs as f16. A subgroup multiplies a 32-row
// region of the tile as products of 16.
// The sums are f32, of f16 terms: close
// to the int8 GEMV's, not equal, and each token's outputs depend only on
// that token, so how a prompt is cut changes no bit.
//
// f16 holds no more than 65504. A token whose largest input passes 2^15,
// which the producer of the input wrote down as the int8 scale of each
// block of 32 (binding 3), has its inputs multiplied by a power of two
// that brings them under it and its outputs by the inverse: exact, short
// of inputs that the scaling pushes below f16's range. For every other
// token the factor is 1.
//
// Bindings and push constants are gemv.wgsl's, so that the host dispatches
// both alike: the batch picks the tokens, output token t starts at
// `t * stride + offset`. The grid is (row tiles, token tiles of the batch's
// capacity); tiles past the pass's tokens return at once.

struct Shape {
    rows: u32,
    blocks_per_row: u32,
    stride: u32,
    offset: u32,
    head: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> xp: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> xd: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> batch: array<u32>;

// The wide tile puts all four subgroups along the row dimension.
const_assert TILE_TOKENS == 64u;
const_assert (TILE_ROWS == 64u || TILE_ROWS == 128u);
const WIDE: bool = TILE_ROWS == 128u;
const THREADS: u32 = 128u;
const WEIGHT_LANES: u32 = THREADS / TILE_ROWS;
const INPUT_LANES: u32 = THREADS / TILE_TOKENS;
const INPUT_VECTORS: u32 = 8u / INPUT_LANES;
// Reuse the output staging for each 16-token strip of the wide tile.
const OUT_TOKENS: u32 = select(TILE_TOKENS, 16u, WIDE);
// Vectors of four f16 per row of a staged block: eight data and two pad
// against bank conflicts. Cooperative loads take strides in vector units.
const LD: u32 = 10u;

// weights [row][k] and inputs [token][k] of one step of 32 columns
var<workgroup> wa: array<vec4<f16>, TILE_ROWS * LD>;
var<workgroup> xb: array<vec4<f16>, TILE_TOKENS * LD>;
// Pad the wide tile's output stride to spread cooperative stores across banks.
const OUT_STRIDE: u32 = TILE_ROWS + select(0u, 4u, WIDE);
var<workgroup> out: array<f32, OUT_TOKENS * OUT_STRIDE>;
// the batch's first token and token count. Every variable here is a
// multiple of 16 bytes: NVIDIA packs them without padding, and the matrix
// loads and stores need addresses aligned to 16
var<workgroup> span: vec4<u32>;
// each staging invocation's largest input, then each token's power of two
var<workgroup> peak: array<f32, THREADS>;
var<workgroup> scale: array<f32, TILE_TOKENS>;


// Element `i` of the f16 plane that starts at word `base`.
fn f16_at(base: u32, i: u32) -> f32 {
    let pair = unpack2x16float(w[base + i / 8u][(i / 2u) % 4u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

fn write_tile(row0: u32, tok0: u32, first: u32, count: u32, tid: u32) {
    for (var i = tid; i < OUT_TOKENS * TILE_ROWS; i += THREADS) {
        let r = i % TILE_ROWS;
        let token = i / TILE_ROWS;
        let row = row0 + r;
        let t = first + token;
        if row < shape.rows && tok0 + t < count {
            y[(tok0 + t) * shape.stride + shape.offset + row] = out[token * OUT_STRIDE + r] * scale[t];
        }
    }
}

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(subgroup_id) slot: u32,
) {
    if tid == 0u {
        var first = 0u;
        var tokens = batch[0];
        if shape.head != 0u {
            first = batch[2];
            tokens = batch[3];
        }
        span = vec4<u32>(first, tokens, 0u, 0u);
    }
    // the same for the whole workgroup, as the matrix products want to know
    let picked = workgroupUniformLoad(&span);
    let start = picked.x;
    let count = picked.y;
    let row0 = group.x * TILE_ROWS;
    let tok0 = group.y * TILE_TOKENS;
    if tok0 >= count {
        return;
    }
    let n = shape.blocks_per_row;
    let total = shape.rows * n;
    let steps = n * SUB;

    // Staging lanes cover a weight row in halves and a token in vectors.
    let wr = tid / WEIGHT_LANES;
    let wh = tid % WEIGHT_LANES;
    let wrow = min(row0 + wr, shape.rows - 1u);
    let weights = weight_row(wrow, shape.rows, n);
    let xr = tid / INPUT_LANES;
    let xh = tid % INPUT_LANES;
    let xrow = start + min(tok0 + xr, count - 1u);

    var sx = 1.0;
    if PACKED {
        if xh == 0u {
            scale[xr] = xd[xrow].x;
        }
    } else {
        // Get the token's power of two from the largest of its block scales.
        var big = 0.0;
        for (var b = xh; b < steps; b += INPUT_LANES) {
            big = max(big, xd[xrow * steps + b].x);
        }
        peak[tid] = big * 127.0;
        workgroupBarrier();
        let top = max(peak[xr * 2u], peak[xr * 2u + 1u]);
        sx = matrix_input_scale(top);
        if xh == 0u {
            scale[xr] = 1.0 / sx;
        }
    }

    let token_quarters = select(2u, 1u, WIDE);
    let qr = (slot / token_quarters) * 32u;
    let qt = (slot % token_quarters) * 32u;
    var c00 = coop_mat16x16<f32, C>();
    var c01 = coop_mat16x16<f32, C>();
    var c10 = coop_mat16x16<f32, C>();
    var c11 = coop_mat16x16<f32, C>();
    var c02 = coop_mat16x16<f32, C>();
    var c03 = coop_mat16x16<f32, C>();
    var c12 = coop_mat16x16<f32, C>();
    var c13 = coop_mat16x16<f32, C>();

    for (var ks = 0u; ks < steps; ks++) {
        let blk = weights.x + (ks / SUB) * weights.y;
        let sub = ks % SUB;
        stage(total, blk, sub, wh, wr * LD + wh * 4u);
        if WEIGHT_LANES == 1u {
            stage(total, blk, sub, 1u, wr * LD + 4u);
        }
        let at = xr * LD + xh * INPUT_VECTORS;
        if PACKED {
            let xi = (ks * count + xrow) * 8u + xh * INPUT_VECTORS;
            for (var j = 0u; j < INPUT_VECTORS; j++) {
                xb[at + j] = xp[xi + j];
            }
        } else {
            let xi = (xrow * steps + ks) * 8u + xh * INPUT_VECTORS;
            for (var j = 0u; j < INPUT_VECTORS; j++) {
                xb[at + j] = vec4<f16>(x[xi + j] * sx);
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 8u; kk += 4u) {
            let a0 = coopLoadT<coop_mat16x16<f16, A>>(&wa[qr * LD + kk], LD);
            let a1 = coopLoadT<coop_mat16x16<f16, A>>(&wa[(qr + 16u) * LD + kk], LD);
            let b0 = coopLoad<coop_mat16x16<f16, B>>(&xb[qt * LD + kk], LD);
            let b1 = coopLoad<coop_mat16x16<f16, B>>(&xb[(qt + 16u) * LD + kk], LD);
            c00 = coopMultiplyAdd(a0, b0, c00);
            c01 = coopMultiplyAdd(a0, b1, c01);
            c10 = coopMultiplyAdd(a1, b0, c10);
            c11 = coopMultiplyAdd(a1, b1, c11);
            if WIDE {
                let b2 = coopLoad<coop_mat16x16<f16, B>>(&xb[32u * LD + kk], LD);
                let b3 = coopLoad<coop_mat16x16<f16, B>>(&xb[48u * LD + kk], LD);
                c02 = coopMultiplyAdd(a0, b2, c02);
                c03 = coopMultiplyAdd(a0, b3, c03);
                c12 = coopMultiplyAdd(a1, b2, c12);
                c13 = coopMultiplyAdd(a1, b3, c13);
            }
        }
        workgroupBarrier();
    }

    // out[token][row]: column-major stores of the row-by-token products
    let left = qt * OUT_STRIDE + qr;
    let right = left + 16u * OUT_STRIDE;
    coopStore(c00, &out[left], OUT_STRIDE);
    coopStore(c10, &out[left + 16u], OUT_STRIDE);
    if OUT_TOKENS == TILE_TOKENS {
        coopStore(c01, &out[right], OUT_STRIDE);
        coopStore(c11, &out[right + 16u], OUT_STRIDE);
    }
    workgroupBarrier();
    write_tile(row0, tok0, 0u, count, tid);
    if WIDE {
        workgroupBarrier();
        coopStore(c01, &out[left], OUT_STRIDE);
        coopStore(c11, &out[left + 16u], OUT_STRIDE);
        workgroupBarrier();
        write_tile(row0, tok0, 16u, count, tid);
        workgroupBarrier();
        coopStore(c02, &out[left], OUT_STRIDE);
        coopStore(c12, &out[left + 16u], OUT_STRIDE);
        workgroupBarrier();
        write_tile(row0, tok0, 32u, count, tid);
        workgroupBarrier();
        coopStore(c03, &out[left], OUT_STRIDE);
        coopStore(c13, &out[left + 16u], OUT_STRIDE);
        workgroupBarrier();
        write_tile(row0, tok0, 48u, count, tid);
    }
}
