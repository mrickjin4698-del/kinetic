/// Output rows and tokens per matrix workgroup. build.rs uses these sizes
/// for the shader variants, each with four subgroups taking 32 output rows each.
pub(crate) const MATRIX_TILE: (usize, usize) = (64, 64);
/// Reuses each token across twice as many weight rows for long passes.
pub(crate) const MATRIX_WIDE_TILE: (usize, usize) = (128, 64);
/// One subgroup's tile for packed inputs to a narrow Q8 projection.
pub(crate) const MATRIX_NARROW_TILE: (usize, usize) = (16, 16);
/// Q8 weight rows interleaved by block column in the Vulkan device layout.
pub(crate) const Q8_WEIGHT_ROWS: usize = 4;
