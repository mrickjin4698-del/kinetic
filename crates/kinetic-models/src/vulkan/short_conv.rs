use kinetic_vulkan::{Commands, ConvRing, DeviceBuffer, DeviceMatrix};

use super::layer::R;
use super::work::{Needs, Work};
use super::Gpu;
use crate::blocks::ShortConv;
use crate::LoadError;

/// The gated short convolution on the device; see [`ShortConv`].
pub(super) struct VulkanShortConv {
    in_proj: DeviceMatrix,
    conv: DeviceBuffer<f32>,
    pub taps: usize,
    out_proj: DeviceMatrix,
}

impl VulkanShortConv {
    pub fn upload(gpu: &Gpu, s: &ShortConv) -> Result<Self, LoadError> {
        Ok(VulkanShortConv {
            in_proj: gpu.matrix(&s.in_proj)?,
            conv: gpu.vector(&s.conv)?,
            taps: s.taps,
            out_proj: gpu.matrix(&s.out_proj)?,
        })
    }

    pub fn size_bytes(&self) -> usize {
        self.in_proj.size_bytes() + self.out_proj.size_bytes()
    }

    /// Convolution channels.
    pub fn channels(&self) -> usize {
        self.out_proj.cols()
    }

    pub fn needs(&self) -> Needs {
        Needs {
            proj: self.in_proj.rows(),
            width: self.channels(),
            rec: 0,
        }
    }

    /// Reads `w.h` and leaves the projected output in `w.y`.
    pub fn record(&self, gpu: &Gpu, w: &Work, ring: &ConvRing, c: &mut Commands) -> R {
        let k = &gpu.kernels;
        let stride = self.in_proj.rows();
        w.project(k, &w.h, &w.proj, stride, &[(&self.in_proj, 0)], c)?;
        let mixed = w.mixed(self.channels());
        k.short_conv(&w.batch, &w.proj, &self.conv, ring, mixed, c)?;
        w.project(k, mixed, &w.y, w.d, &[(&self.out_proj, 0)], c)
    }
}
