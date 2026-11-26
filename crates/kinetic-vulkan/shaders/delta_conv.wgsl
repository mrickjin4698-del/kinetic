@group(0) @binding(2) var<storage, read> weight: array<f32>;

fn convolve(t: u32, pos0: u32, ch: u32, x: f32) -> f32 {
    let pos = pos0 + t;
    let w = ch * shape.taps;
    var past = 0.0;
    for (var j = 0u; j + 1u < shape.taps; j++) {
        let back = shape.taps - 1u - j;
        if pos >= back {
            past += weight[w + j] * input(pos - back, pos0, ch);
        }
    }
    return weight[w + shape.taps - 1u] * x + past;
}
