// The input must fit into f16. Scale by a power of two so the matrix
// product can apply its inverse to the f32 outputs.
fn matrix_input_scale(peak: f32) -> f32 {
    const SAFE: f32 = 32768.0;
    if peak > SAFE {
        return ldexp(1.0, -i32(ceil(log2(peak / SAFE))));
    }
    return 1.0;
}
