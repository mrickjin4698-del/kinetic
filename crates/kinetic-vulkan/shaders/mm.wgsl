enable f16;
enable wgpu_cooperative_matrix;

// Shared part of every kernel on the matrix units; build.rs appends one
// weight format to it, which defines `SUB`, the steps of 32 columns per
// block, and `stage`, which expands 16 weights of a step to f16.
//
// Y = W X for a pass of many tokens, on the matrix units of devices with
// cooperative 16x16x16 f16 products into f32. A workgroup of four
// subgroups takes 64 rows of W by 64 tokens; per step of 32 columns it
// expands the weights to f16 in workgroup memory, and the f32 inputs to
// f16 beside them, and each subgroup multiplies a 32 by 32 quarter of the
// tile as two by two products of 16. The sums are f32, of f16 terms: close
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
@group(0) @binding(2) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> xd: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> batch: array<u32>;

// TILE, the rows and tokens of a workgroup, comes from build.rs; the
// subgroup quarters and the staging below are laid out for 64.
const_assert TILE == 64u;
// Vectors of four f16 per row of a staged block: eight data and two pad
// against bank conflicts. Cooperative loads take strides in vector units.
const LD: u32 = 10u;

// weights [row][k] and inputs [token][k] of one step of 32 columns
var<workgroup> wa: array<vec4<f16>, TILE * LD>;
var<workgroup> xb: array<vec4<f16>, TILE * LD>;
// the finished tile, [token][row]
var<workgroup> out: array<f32, TILE * TILE>;
// the batch's first token and token count. Every variable here is a
// multiple of 16 bytes: NVIDIA packs them without padding, and the matrix
// loads and stores need addresses aligned to 16
var<workgroup> span: vec4<u32>;
// each staging invocation's largest input, then each token's power of two
var<workgroup> peak: array<f32, 2u * TILE>;
var<workgroup> scale: array<f32, TILE>;

// Inputs above this get scaled down.
const SAFE: f32 = 32768.0;

// Element `i` of the f16 plane that starts at word `base`.
fn f16_at(base: u32, i: u32) -> f32 {
    let pair = unpack2x16float(w[base + i / 8u][(i / 2u) % 4u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
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
    let row0 = group.x * TILE;
    let tok0 = group.y * TILE;
    if tok0 >= count {
        return;
    }
    let n = shape.blocks_per_row;
    let total = shape.rows * n;
    let steps = n * SUB;

    // staging: invocation tid takes half `half` of row (and token) tid / 2
    let r = tid / 2u;
    let half = tid % 2u;
    let wrow = min(row0 + r, shape.rows - 1u);
    let xrow = start + min(tok0 + r, count - 1u);

    // this subgroup's quarter: rows qr.., tokens qt..
    // the token's power of two, from the largest of its block scales
    var big = 0.0;
    for (var b = half; b < steps; b += 2u) {
        big = max(big, xd[xrow * steps + b].x);
    }
    peak[tid] = big * 127.0;
    workgroupBarrier();
    let top = max(peak[r * 2u], peak[r * 2u + 1u]);
    var sx = 1.0;
    if top > SAFE {
        sx = ldexp(1.0, -i32(ceil(log2(top / SAFE))));
    }
    if half == 0u {
        scale[r] = 1.0 / sx;
    }

    let qr = (slot / 2u) * 32u;
    let qt = (slot % 2u) * 32u;
    var c00 = coop_mat16x16<f32, C>();
    var c01 = coop_mat16x16<f32, C>();
    var c10 = coop_mat16x16<f32, C>();
    var c11 = coop_mat16x16<f32, C>();

    for (var ks = 0u; ks < steps; ks++) {
        let blk = wrow * n + ks / SUB;
        let at = r * LD + half * 4u;
        stage(total, blk, ks % SUB, half, at);
        let xi = xrow * steps * 8u + ks * 8u + half * 4u;
        for (var j = 0u; j < 4u; j++) {
            xb[at + j] = vec4<f16>(x[xi + j] * sx);
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
        }
        workgroupBarrier();
    }

    // out[token][row]: column-major stores of the row-by-token products
    coopStore(c00, &out[qt * TILE + qr], TILE);
    coopStore(c01, &out[(qt + 16u) * TILE + qr], TILE);
    coopStore(c10, &out[qt * TILE + qr + 16u], TILE);
    coopStore(c11, &out[(qt + 16u) * TILE + qr + 16u], TILE);
    workgroupBarrier();
    for (var i = tid; i < TILE * TILE; i += 128u) {
        let row = row0 + i % TILE;
        let t = tok0 + i / TILE;
        if row < shape.rows && t < count {
            y[t * shape.stride + shape.offset + row] = out[i] * scale[i / TILE];
        }
    }
}
