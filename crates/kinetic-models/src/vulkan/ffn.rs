use kinetic_vulkan::{Commands, DeviceMatrix};

use super::layer::{unsupported, R};
use super::work::Work;
use super::Gpu;
use crate::blocks::{Activation, SwiGlu};
use crate::LoadError;

/// The gated feed-forward on the device; see [`SwiGlu`].
pub(super) struct VulkanFfn {
    pub gate: DeviceMatrix,
    up: DeviceMatrix,
    down: DeviceMatrix,
}

impl VulkanFfn {
    pub fn upload(gpu: &Gpu, f: &SwiGlu) -> Result<Self, LoadError> {
        if f.activation != Activation::Silu {
            return Err(unsupported("feed-forward activations other than SiLU"));
        }
        Ok(VulkanFfn {
            gate: gpu.matrix(&f.gate)?,
            up: gpu.matrix(&f.up)?,
            down: gpu.matrix(&f.down)?,
        })
    }

    pub fn size_bytes(&self) -> usize {
        self.gate.size_bytes() + self.up.size_bytes() + self.down.size_bytes()
    }

    /// Reads `w.h` and leaves the output in `w.y`.
    pub fn record(&self, gpu: &Gpu, w: &Work, c: &mut Commands) -> R {
        let k = &gpu.kernels;
        let n_ff = self.gate.rows();
        w.project(
            k,
            &w.h,
            &w.gu,
            2 * n_ff,
            &[(&self.gate, 0), (&self.up, n_ff)],
            c,
        )?;
        let act = w.activations(n_ff);
        k.swiglu(&w.batch, &w.gu, act, c)?;
        w.project(k, act, &w.y, w.d, &[(&self.down, 0)], c)
    }
}
