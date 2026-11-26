@group(0) @binding(2) var<storage, read> weight: array<vec4<f32>>;

// All but the first three rows read their entire convolution from this pass.
fn convolve(t: u32, pos0: u32, ch: u32, x: f32) -> f32 {
    let w = weight[ch];
    var past = 0.0;
    if t >= 3u {
        let stride = shape.stride;
        let first = (t - 3u) * stride + ch;
        past += w.x * proj[first];
        past += w.y * proj[first + stride];
        past += w.z * proj[first + 2u * stride];
    } else {
        let pos = pos0 + t;
        if pos >= 3u {
            past += w.x * input(pos - 3u, pos0, ch);
        }
        if pos >= 2u {
            past += w.y * input(pos - 2u, pos0, ch);
        }
        if pos >= 1u {
            past += w.z * input(pos - 1u, pos0, ch);
        }
    }
    return w.w * x + past;
}
