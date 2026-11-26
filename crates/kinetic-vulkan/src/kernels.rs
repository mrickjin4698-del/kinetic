use std::rc::Rc;

#[cfg(not(metal_driver))]
use kinetic_quant::TensorType;

use crate::commands::Commands;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::memory::DeviceBuffer;
use crate::pipeline::Kernel;

/// A kernel as build.rs wrote it for the driver: SPIR-V for Vulkan, Metal
/// shader source for Metal.
#[cfg(not(metal_driver))]
macro_rules! spirv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}
#[cfg(metal_driver)]
macro_rules! spirv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metal"))
    };
}

const READ_SUM_THREADS: u64 = 256;
/// 16-byte words each read_sum invocation reads, at least; more workgroups
/// would only add scheduling overhead.
const READ_SUM_WORDS_PER_THREAD: u64 = 16;
/// The dispatch size every device supports in each dimension.
pub(crate) const MAX_GROUPS: u32 = 65535;

/// The crate's kernels, built for one device.
pub struct Kernels {
    pub(crate) gemv_f32: Rc<Kernel>,
    pub(crate) gemv_f16: Rc<Kernel>,
    pub(crate) gemv_q8_0: Rc<Kernel>,
    pub(crate) gemv_q4_k: Rc<Kernel>,
    /// The same sums for a pass of many tokens; see `shaders/metal`.
    #[cfg(metal_driver)]
    pub(crate) gemv_q4_k_many: Rc<Kernel>,
    #[cfg(metal_driver)]
    pub(crate) gemv_q5_k_many: Rc<Kernel>,
    #[cfg(metal_driver)]
    pub(crate) gemv_q6_k_many: Rc<Kernel>,
    pub(crate) gemv_q5_k: Rc<Kernel>,
    pub(crate) gemv_q6_k: Rc<Kernel>,
    /// Products of prompt passes on the matrix units, on devices with
    /// cooperative f16 matrices (`Features::matrix_f16`).
    #[cfg(not(metal_driver))]
    pub(crate) mm: Vec<(TensorType, [Rc<Kernel>; 3])>,
    #[cfg(not(metal_driver))]
    pub(crate) mm_pack: Option<Rc<Kernel>>,
    pub(crate) quantize_q8: Rc<Kernel>,
    pub(crate) norm: Rc<Kernel>,
    pub(crate) swiglu: Rc<Kernel>,
    pub(crate) argmax: Rc<Kernel>,
    pub(crate) rope: Rc<Kernel>,
    pub(crate) short_conv: Rc<Kernel>,
    pub(crate) delta_conv: Rc<Kernel>,
    pub(crate) delta_rule: Rc<Kernel>,
    pub(crate) gated_norm: Rc<Kernel>,
    pub(crate) attend_64: Rc<Kernel>,
    pub(crate) attend_128: Rc<Kernel>,
    pub(crate) attend_256: Rc<Kernel>,
    read_sum: Rc<Kernel>,
    /// The batch words of a pass of one token, for GEMVs outside a model.
    pub(crate) single: DeviceBuffer<u32>,
}

/// The weight types with kernels on the matrix units, and those kernels,
/// as build.rs compiles them.
#[cfg(not(metal_driver))]
const MATRIX_TYPES: [(TensorType, &str, [&[u8]; 3]); 4] = [
    (
        TensorType::Q8_0,
        "mm_q8_0",
        [
            spirv!("mm_q8_0"),
            spirv!("mm_q8_0_wide"),
            spirv!("mm_q8_0_packed"),
        ],
    ),
    (
        TensorType::Q4K,
        "mm_q4_k",
        [
            spirv!("mm_q4_k"),
            spirv!("mm_q4_k_wide"),
            spirv!("mm_q4_k_packed"),
        ],
    ),
    (
        TensorType::Q5K,
        "mm_q5_k",
        [
            spirv!("mm_q5_k"),
            spirv!("mm_q5_k_wide"),
            spirv!("mm_q5_k_packed"),
        ],
    ),
    (
        TensorType::Q6K,
        "mm_q6_k",
        [
            spirv!("mm_q6_k"),
            spirv!("mm_q6_k_wide"),
            spirv!("mm_q6_k_packed"),
        ],
    ),
];

/// Most tokens of a prompt pass on the matrix units; every state holds
/// buffers for as many.
#[cfg(not(metal_driver))]
const MATRIX_PROMPT_BATCH: usize = 512;

impl Kernels {
    /// Builds the pipelines for the kernels embedded in this crate.
    pub fn load(device: &Rc<Device>) -> Result<Kernels> {
        Kernels::load_with(device, true)
    }

    /// [`Kernels::load`], with prompt passes on the matrix units only if
    /// `matrix` and the device has them; see [`Kernels::matrix`].
    pub fn load_with(device: &Rc<Device>, matrix: bool) -> Result<Kernels> {
        let gemv = |name: &str, spirv| Kernel::new(device, name, spirv, 6, 5);
        let attend = |name, spirv| Kernel::new(device, name, spirv, 7, 9);
        #[cfg(not(metal_driver))]
        let matrix = matrix && device.info().features.matrix_f16;
        #[cfg(metal_driver)]
        let _ = matrix;
        Ok(Kernels {
            gemv_f32: gemv("gemv_f32", spirv!("gemv_f32"))?,
            gemv_f16: gemv("gemv_f16", spirv!("gemv_f16"))?,
            gemv_q8_0: gemv("gemv_q8_0", spirv!("gemv_q8_0"))?,
            gemv_q4_k: gemv("gemv_q4_k", spirv!("gemv_q4_k"))?,
            #[cfg(metal_driver)]
            gemv_q4_k_many: gemv(
                "gemv_q4_k_many",
                include_bytes!("../shaders/metal/gemv_q4_k_many.metal"),
            )?,
            #[cfg(metal_driver)]
            gemv_q5_k_many: gemv(
                "gemv_q5_k_many",
                include_bytes!("../shaders/metal/gemv_q5_k_many.metal"),
            )?,
            #[cfg(metal_driver)]
            gemv_q6_k_many: gemv(
                "gemv_q6_k_many",
                include_bytes!("../shaders/metal/gemv_q6_k_many.metal"),
            )?,
            gemv_q5_k: gemv("gemv_q5_k", spirv!("gemv_q5_k"))?,
            gemv_q6_k: gemv("gemv_q6_k", spirv!("gemv_q6_k"))?,
            #[cfg(not(metal_driver))]
            mm: match matrix {
                true => MATRIX_TYPES
                    .iter()
                    .map(|&(ty, name, [regular, wide, packed])| {
                        Ok((
                            ty,
                            [
                                gemv(name, regular)?,
                                gemv(&format!("{name}_wide"), wide)?,
                                gemv(&format!("{name}_packed"), packed)?,
                            ],
                        ))
                    })
                    .collect::<Result<_>>()?,
                false => Vec::new(),
            },
            #[cfg(not(metal_driver))]
            mm_pack: matrix
                .then(|| Kernel::new(device, "mm_pack", spirv!("mm_pack"), 5, 1))
                .transpose()?,
            quantize_q8: Kernel::new(device, "quantize_q8", spirv!("quantize_q8"), 3, 1)?,
            norm: Kernel::new(device, "norm", spirv!("norm"), 8, 4)?,
            swiglu: Kernel::new(device, "swiglu", spirv!("swiglu"), 5, 1)?,
            argmax: Kernel::new(device, "argmax", spirv!("argmax"), 3, 1)?,
            rope: Kernel::new(device, "rope", spirv!("rope"), 5, 13)?,
            short_conv: Kernel::new(device, "short_conv", spirv!("short_conv"), 7, 3)?,
            delta_conv: Kernel::new(device, "delta_conv", spirv!("delta_conv"), 6, 7)?,
            delta_rule: Kernel::new(device, "delta_rule", spirv!("delta_rule"), 4, 7)?,
            gated_norm: Kernel::new(device, "gated_norm", spirv!("gated_norm"), 7, 5)?,
            attend_64: attend("attend_64", spirv!("attend_64"))?,
            attend_128: attend("attend_128", spirv!("attend_128"))?,
            attend_256: attend("attend_256", spirv!("attend_256"))?,
            read_sum: Kernel::new(device, "read_sum", spirv!("read_sum"), 2, 1)?,
            single: DeviceBuffer::from_slice(device, &[1, 0, 0, 1])?,
        })
    }

    /// Whether prompt passes run their products on the matrix units, in
    /// f16 terms: close to the GEMV's sums, not equal; see
    /// [`Kernels::exact_prompt`].
    pub fn matrix(&self) -> bool {
        #[cfg(metal_driver)]
        return false;
        #[cfg(not(metal_driver))]
        !self.mm.is_empty()
    }

    /// Most tokens worth running as one prompt pass, or 0 where the device
    /// has no kernels of its own for such passes; see
    /// [`Commands::set_prompt`].
    pub fn prompt_batch(&self) -> usize {
        #[cfg(metal_driver)]
        return 64;
        #[cfg(not(metal_driver))]
        match self.mm.is_empty() {
            false => MATRIX_PROMPT_BATCH,
            true => 0,
        }
    }

    /// Reads all of `data` once and adds its sum as 32-bit words to `out[0]`.
    ///
    /// The sum is meaningless; the kernel's throughput is the achievable
    /// read bandwidth of device memory, the ceiling for memory-bound kernels.
    /// `data.len()` must be a multiple of 16.
    pub fn read_sum(
        &self,
        data: &DeviceBuffer<u8>,
        out: &DeviceBuffer<u32>,
        commands: &mut Commands,
    ) -> Result<()> {
        if !data.len().is_multiple_of(16) || out.is_empty() {
            return Err(Error::Shape(format!(
                "read_sum needs a multiple of 16 bytes and one output, got {} and {}",
                data.len(),
                out.len()
            )));
        }
        let words = (data.len() / 16) as u64;
        let groups = words
            .div_ceil(READ_SUM_THREADS * READ_SUM_WORDS_PER_THREAD)
            .clamp(1, MAX_GROUPS.into()) as u32;
        // SAFETY: matches read_sum.wgsl: `data` and `out` at bindings 0 and
        // 1 and the word count, which fits in 32 bits because buffers are
        // capped at the device's storage buffer range.
        unsafe {
            commands.dispatch(
                &self.read_sum,
                &[data.allocation(), out.allocation()],
                &[words as u32],
                [groups, 1, 1],
            )
        };
        Ok(())
    }
}
