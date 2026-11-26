const DK: u32 = 128u;

// Keep the same four consecutive elements per lane in both recurrence paths.
fn delta_norm_scale(x: vec4<f32>, eps: f32) -> f32 {
    let norm = sqrt(subgroupAdd(dot(x, x)));
    return 1.0 / max(norm, eps);
}
