// Copy two planes of the batch's selected input row to row zero.
// Read and write words to preserve the floats and packed int8 values.
struct Shape {
    first: u32,
    second: u32,
}

var<immediate> shape: Shape;
@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read> src0: array<u32>;
@group(0) @binding(2) var<storage, read> src1: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst0: array<u32>;
@group(0) @binding(4) var<storage, read_write> dst1: array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if batch[3] != 1u {
        return;
    }
    let row = batch[2];
    let i = id.x;
    if i < shape.first {
        dst0[i] = src0[row * shape.first + i];
    }
    if i < shape.second {
        dst1[i] = src1[row * shape.second + i];
    }
}
