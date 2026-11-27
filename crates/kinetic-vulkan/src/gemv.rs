use std::rc::Rc;

use kinetic_quant::{Matrix, TensorType};

use crate::batch::Batch;
use crate::commands::Commands;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::kernels::{Kernels, MAX_GROUPS};
use crate::layout::device_layout;
use crate::matrix_scratch::MatrixScratch;
#[cfg(not(metal_driver))]
use crate::matrix_tile::{MATRIX_NARROW_TILE, MATRIX_TILE, MATRIX_WIDE_TILE};
use crate::memory::{Allocation, DeviceBuffer};
use crate::pipeline::Kernel;

/// Rows per workgroup; each row gets one subgroup of 32.
#[cfg(not(metal_driver))]
const ROWS_PER_GROUP: usize = 4;
/// Rows per workgroup of the GEMV kernels in `shaders/metal`: four
/// subgroups of 32, each taking a tile of four rows.
#[cfg(metal_driver)]
const ROWS_PER_GROUP: usize = 16;
/// Rows per workgroup of the kernels for many tokens in `shaders/metal`:
/// four subgroups of two rows.
#[cfg(metal_driver)]
const MANY_ROWS_PER_GROUP: usize = 8;
/// Elements per scale in the int8 copy of the input.
const Q8_BLOCK: usize = 32;
/// Invocations per workgroup of quantize_q8.wgsl, one per block of 32.
const QUANTIZE_THREADS: usize = 64;
/// Invocations per workgroup of head_input.wgsl.
const HEAD_INPUT_THREADS: usize = 128;

/// Elements per kernel block for each type with a GEMV kernel.
fn block_len(ty: TensorType) -> Option<usize> {
    Some(match ty {
        TensorType::F32 => 4,
        TensorType::F16 => 8,
        TensorType::Q8_0 => 32,
        TensorType::Q4K | TensorType::Q5K | TensorType::Q6K => 256,
        _ => return None,
    })
}

/// A weight matrix in device memory, laid out for the GEMV kernels.
pub struct DeviceMatrix {
    ty: TensorType,
    rows: usize,
    cols: usize,
    bytes: usize,
    data: DeviceBuffer<u8>,
}

impl DeviceMatrix {
    /// Copies `m` to the device. Fails for types without a GEMV kernel and
    /// for row lengths that are not a multiple of 32.
    pub fn upload(device: &Rc<Device>, m: &Matrix) -> Result<DeviceMatrix> {
        let Some(block) = block_len(m.ty) else {
            return Err(Error::Unsupported(m.ty));
        };
        let blocks = m.rows.checked_mul(m.cols / block);
        if !m.cols.is_multiple_of(Q8_BLOCK) || blocks.is_none_or(|b| u32::try_from(b).is_err()) {
            return Err(Error::Shape(format!(
                "{}x{} matrix: columns must be a multiple of {Q8_BLOCK} and blocks fit in 32 bits",
                m.rows, m.cols
            )));
        }
        let mut data = device_layout(m).into_owned();
        // Kernels read whole 16-byte words, also of the last plane.
        data.resize(data.len().next_multiple_of(16), 0);
        Ok(DeviceMatrix {
            ty: m.ty,
            rows: m.rows,
            cols: m.cols,
            bytes: m.data.len(),
            data: DeviceBuffer::from_slice(device, &data)?,
        })
    }

    pub fn ty(&self) -> TensorType {
        self.ty
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes of weights, which a GEMV reads once.
    pub fn size_bytes(&self) -> usize {
        self.bytes
    }
}

/// Rows of GEMV input vectors, kept both as f32 for float weights and
/// quantized to int8 in blocks of 32 for quantized weights.
pub struct GemvInput {
    width: usize,
    x: DeviceBuffer<f32>,
    qs: DeviceBuffer<i8>,
    ds: DeviceBuffer<f32>,
}

impl GemvInput {
    /// One input of `len` elements; `len` must be a multiple of 32.
    pub fn new(device: &Rc<Device>, len: usize) -> Result<GemvInput> {
        GemvInput::batch(device, len, 1)
    }

    /// `rows` inputs of `width` elements each; `width` must be a multiple
    /// of 32.
    pub fn batch(device: &Rc<Device>, width: usize, rows: usize) -> Result<GemvInput> {
        let len = width
            .checked_mul(rows)
            .filter(|&l| u32::try_from(l).is_ok());
        let Some(len) = len.filter(|_| width.is_multiple_of(Q8_BLOCK)) else {
            return Err(Error::Shape(format!(
                "GEMV input of {rows} rows of {width} is not a 32-bit multiple of {Q8_BLOCK}"
            )));
        };
        Ok(GemvInput {
            width,
            x: DeviceBuffer::new(device, len)?,
            qs: DeviceBuffer::new(device, len)?,
            ds: DeviceBuffer::new(device, 2 * len / Q8_BLOCK)?,
        })
    }

    /// Elements of one input.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Inputs held.
    pub fn rows(&self) -> usize {
        self.x.len() / self.width.max(1)
    }

    /// All elements of all rows.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// The f32 values, row after row.
    pub fn values(&self) -> &DeviceBuffer<f32> {
        &self.x
    }

    /// The three buffers a kernel writes the input to, at consecutive
    /// bindings in this order.
    pub(crate) fn outputs(&self) -> [&Rc<Allocation>; 3] {
        [
            self.x.allocation(),
            self.qs.allocation(),
            self.ds.allocation(),
        ]
    }

    /// Copies `x` to the device now and records its quantization in
    /// `commands`, which runs at submit: a second upload before then
    /// replaces the first.
    pub fn upload(&mut self, kernels: &Kernels, x: &[f32], commands: &mut Commands) -> Result<()> {
        self.x.copy_from(x)?;
        self.record_quantize(kernels, commands)
    }

    /// Records the quantization of every row from the f32 values.
    pub fn record_quantize(&self, kernels: &Kernels, commands: &mut Commands) -> Result<()> {
        let blocks = self.x.len() / Q8_BLOCK;
        let groups = blocks.div_ceil(QUANTIZE_THREADS);
        if groups > MAX_GROUPS as usize {
            return Err(Error::Shape(format!(
                "GEMV input of {} elements is too long to quantize",
                self.x.len()
            )));
        }
        // SAFETY: matches quantize_q8.wgsl: x, qs and ds at bindings 0..2
        // and the block count; it touches `blocks` blocks of each.
        unsafe {
            commands.dispatch(
                &kernels.quantize_q8,
                &[
                    self.x.allocation(),
                    self.qs.allocation(),
                    self.ds.allocation(),
                ],
                &[blocks as u32],
                [groups as u32, 1, 1],
            )
        };
        Ok(())
    }

    /// The int8 values: element `i` is `quantized[i] * d` of its block.
    pub fn quantized(&self) -> &DeviceBuffer<i8> {
        &self.qs
    }

    /// Per block of 32, interleaved: the scale `d` and `d` times the sum of
    /// the block's int8 values.
    pub fn block_scales(&self) -> &DeviceBuffer<f32> {
        &self.ds
    }
}

/// Which input rows a batched GEMV multiplies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    /// Every token of the batch.
    Tokens,
    /// The rows the output head runs on; see [`Batch::set`].
    Head,
}

/// Where a batched GEMV writes: output row `t` starts at element
/// `t * stride + offset` of the output buffer.
#[derive(Clone, Copy, Debug)]
pub struct Out {
    pub stride: usize,
    pub offset: usize,
}

impl Kernels {
    /// Copies a batch's single head row to row zero of smaller buffers.
    /// Both the residual and the three normalized input planes keep their bits.
    pub fn copy_head_input(
        &self,
        batch: &Batch,
        residual: &DeviceBuffer<f32>,
        input: &GemvInput,
        out_residual: &DeviceBuffer<f32>,
        out: &GemvInput,
        commands: &mut Commands,
    ) -> Result<()> {
        let width = input.width();
        let fits = width != 0
            && out.width() == width
            && input.rows() >= batch.rows()
            && residual.len() / width >= batch.rows()
            && out.rows() != 0
            && out_residual.len() >= width
            && width.div_ceil(HEAD_INPUT_THREADS) <= MAX_GROUPS as usize;
        let [x, qs, ds] = input.outputs();
        let [ox, oqs, ods] = out.outputs();
        let src = [residual.allocation(), x, qs, ds];
        let dst = [out_residual.allocation(), ox, oqs, ods];
        let distinct = dst
            .iter()
            .enumerate()
            .all(|(i, &a)| !src.iter().chain(&dst[..i]).any(|&b| Rc::ptr_eq(a, b)));
        if !fits || !distinct {
            return Err(Error::Shape(format!(
                "head input of {width} elements and {} rows",
                batch.rows()
            )));
        }
        commands.independent(|commands| {
            for (i, [first, second]) in [[width, width], [width / 4, width / Q8_BLOCK * 2]]
                .into_iter()
                .enumerate()
            {
                let at = i * 2;
                // SAFETY: the batch selects a row within its checked capacity.
                // Each dispatch copies two planes into distinct destination
                // buffers, with lengths in four-byte words.
                unsafe {
                    commands.dispatch(
                        &self.head_input,
                        &[
                            batch.device().allocation(),
                            src[at],
                            src[at + 1],
                            dst[at],
                            dst[at + 1],
                        ],
                        &[first as u32, second as u32],
                        [first.max(second).div_ceil(HEAD_INPUT_THREADS) as u32, 1, 1],
                    )
                };
            }
            Ok(())
        })
    }

    /// Records `y = w x` in `commands`.
    pub fn gemv(
        &self,
        w: &DeviceMatrix,
        x: &GemvInput,
        y: &DeviceBuffer<f32>,
        commands: &mut Commands,
    ) -> Result<()> {
        if x.len() != w.cols || y.len() != w.rows {
            return Err(Error::Shape(format!(
                "GEMV of a {}x{} matrix with {} inputs into {} outputs",
                w.rows,
                w.cols,
                x.len(),
                y.len()
            )));
        }
        let out = Out {
            stride: w.rows,
            offset: 0,
        };
        // SAFETY: the shapes were checked above, and the batch of one
        // input selects its only row either way.
        unsafe { self.record_gemv(w, x, y, out, false, false, &self.single, commands) }
    }

    /// Records `y = w x` for the rows of `x` that `batch` and `rows`
    /// choose, written as `out` says.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_rows(
        &self,
        w: &DeviceMatrix,
        x: &GemvInput,
        y: &DeviceBuffer<f32>,
        out: Out,
        rows: Rows,
        batch: &Batch,
        commands: &mut Commands,
    ) -> Result<()> {
        let capacity = match rows {
            Rows::Tokens => batch.rows(),
            Rows::Head => batch.head_rows(),
        };
        Self::check_projection(w, x, y, out, capacity, batch)?;
        self.record_rows(w, x, y, out, rows, batch, None, commands)
    }

    /// Records projections of the token rows of `x` into disjoint slices
    /// of each output row. With scratch, long matrix passes pack `x` once
    /// for all the projections that run on the matrix units.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_projections(
        &self,
        projections: &[(&DeviceMatrix, usize)],
        x: &GemvInput,
        y: &DeviceBuffer<f32>,
        stride: usize,
        batch: &Batch,
        scratch: Option<&MatrixScratch>,
        commands: &mut Commands,
    ) -> Result<()> {
        if projections.is_empty() {
            return Ok(());
        }
        let capacity = batch.rows();
        for (i, &(w, offset)) in projections.iter().enumerate() {
            Self::check_projection(w, x, y, Out { stride, offset }, capacity, batch)?;
            let end = offset + w.rows;
            if projections[i + 1..]
                .iter()
                .any(|&(other, at)| at < end && offset < at.saturating_add(other.rows))
            {
                return Err(Error::Shape("overlapping projection outputs".into()));
            }
        }

        #[cfg(not(metal_driver))]
        let prepared = match (&self.mm_pack, scratch) {
            (Some(kernel), Some(scratch))
                if commands.prompt()
                    && capacity > MATRIX_TILE.1
                    && capacity <= MAX_GROUPS as usize
                    && capacity * (x.width() / 2) <= scratch.values.len()
                    && 2 * capacity <= scratch.scales.len()
                    && projections
                        .iter()
                        .any(|(w, _)| self.matrix_kernels(w.ty).is_some()) =>
            {
                // SAFETY: x holds all rows in this batch and scratch holds
                // their half-precision values plus two scale words per row.
                // mm_pack reads the batch count and stops beyond it.
                unsafe {
                    commands.dispatch(
                        kernel,
                        &[
                            batch.device().allocation(),
                            x.x.allocation(),
                            x.ds.allocation(),
                            scratch.values.allocation(),
                            scratch.scales.allocation(),
                        ],
                        &[(x.width() / 32) as u32],
                        [capacity as u32, 1, 1],
                    )
                };
                Some(scratch)
            }
            _ => None,
        };
        #[cfg(metal_driver)]
        let prepared = {
            let _ = scratch;
            None
        };

        commands.independent(|commands| {
            for &(w, offset) in projections {
                self.record_rows(
                    w,
                    x,
                    y,
                    Out { stride, offset },
                    Rows::Tokens,
                    batch,
                    prepared,
                    commands,
                )?;
            }
            Ok(())
        })
    }

    fn check_projection(
        w: &DeviceMatrix,
        x: &GemvInput,
        y: &DeviceBuffer<f32>,
        out: Out,
        capacity: usize,
        batch: &Batch,
    ) -> Result<()> {
        let end = out.offset.checked_add(w.rows);
        let needed = (capacity.max(1) - 1)
            .checked_mul(out.stride)
            .and_then(|start| end.and_then(|end| start.checked_add(end)));
        if x.width() != w.cols
            || x.rows() < batch.rows()
            || needed.is_none_or(|n| n > y.len() || u32::try_from(n).is_err())
            || (capacity > 1 && end.is_some_and(|end| out.stride < end))
        {
            return Err(Error::Shape(format!(
                "GEMV of a {}x{} matrix on {} rows of {} into {} outputs at stride {} offset {}",
                w.rows,
                w.cols,
                x.rows(),
                x.width(),
                y.len(),
                out.stride,
                out.offset
            )));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_rows(
        &self,
        w: &DeviceMatrix,
        x: &GemvInput,
        y: &DeviceBuffer<f32>,
        out: Out,
        rows: Rows,
        batch: &Batch,
        prepared: Option<&MatrixScratch>,
        commands: &mut Commands,
    ) -> Result<()> {
        // A pass of many prompt tokens runs every row of the batch.
        let many = rows == Rows::Tokens && commands.prompt();
        #[cfg(metal_driver)]
        let _ = prepared;
        #[cfg(not(metal_driver))]
        if many {
            if let Some(kernels) = self.matrix_kernels(w.ty) {
                let capacity = batch.rows();
                let (kernel, tile, input) = match prepared {
                    Some(scratch) => {
                        use crate::cooperative_matrix2::TILE;
                        // The CM2 tile reuses packed inputs for Q8 projections
                        // with a long product dimension, such as FFN down.
                        let (kernel, tile) = match (&self.mm_q8_0_cm2, &self.mm_q8_0_packed_narrow)
                        {
                            (Some(cm2), _)
                                if w.ty == TensorType::Q8_0
                                    && w.rows.is_multiple_of(TILE.0)
                                    && w.cols >= w.rows
                                    && w.cols >= 2048
                                    && capacity >= TILE.1 =>
                            {
                                (cm2, TILE)
                            }
                            (_, Some(narrow)) if w.ty == TensorType::Q8_0 && w.rows <= 32 => {
                                (narrow, MATRIX_NARROW_TILE)
                            }
                            _ => (&kernels[2], MATRIX_WIDE_TILE),
                        };
                        (
                            kernel,
                            tile,
                            [
                                x.x.allocation(),
                                scratch.values.allocation(),
                                scratch.scales.allocation(),
                            ],
                        )
                    }
                    None if capacity > MATRIX_TILE.1 => {
                        (&kernels[1], MATRIX_WIDE_TILE, x.outputs())
                    }
                    None => (&kernels[0], MATRIX_TILE, x.outputs()),
                };
                // SAFETY: as below, with f16 blocks and per-token scales when
                // prepare wrote them, and the original input buffers otherwise.
                unsafe {
                    self.record_matrix(kernel, w, input, y, out, capacity, tile, batch, commands)
                };
                return Ok(());
            }
        }
        // SAFETY: checked above: `x` holds every row the batch can name and
        // `y` every output row it can ask for.
        unsafe {
            self.record_gemv(
                w,
                x,
                y,
                out,
                rows == Rows::Head,
                many,
                batch.device(),
                commands,
            )
        }
    }

    /// # Safety
    ///
    /// `x` must hold every input row and `y` every output row that the
    /// words in `batch` select.
    #[allow(clippy::too_many_arguments)]
    unsafe fn record_gemv(
        &self,
        w: &DeviceMatrix,
        x: &GemvInput,
        y: &DeviceBuffer<f32>,
        out: Out,
        head: bool,
        many: bool,
        batch: &DeviceBuffer<u32>,
        commands: &mut Commands,
    ) -> Result<()> {
        let block = block_len(w.ty).ok_or(Error::Unsupported(w.ty))?;
        let (kernel, rows_per_group) = self.gemv_kernel(w.ty, many);
        let groups = w.rows.div_ceil(rows_per_group);
        let wide = groups.min(MAX_GROUPS as usize);
        let tall = groups.div_ceil(wide.max(1));
        // SAFETY: matches gemv.wgsl: weights, x, qs, ds, y and the batch at
        // bindings 0..5, then rows, blocks per row, output stride and
        // offset, and the choice of rows. The kernel reads `rows * cols`
        // weights and the input rows the batch selects and writes the
        // output rows it selects, which the caller vouches for; workgroups
        // past the last row return early.
        unsafe {
            commands.dispatch(
                kernel,
                &[
                    w.data.allocation(),
                    x.x.allocation(),
                    x.qs.allocation(),
                    x.ds.allocation(),
                    y.allocation(),
                    batch.allocation(),
                ],
                &[
                    w.rows as u32,
                    (w.cols / block) as u32,
                    out.stride as u32,
                    out.offset as u32,
                    head as u32,
                ],
                [wide as u32, tall as u32, 1],
            )
        };
        Ok(())
    }

    /// Whether prompt passes with weights of type `ty` give every token the
    /// bits of the GEMV, which they don't where they run on the matrix units.
    pub fn exact_prompt(&self, ty: TensorType) -> bool {
        #[cfg(not(metal_driver))]
        {
            self.matrix_kernels(ty).is_none()
        }
        #[cfg(metal_driver)]
        {
            let _ = ty;
            true
        }
    }

    /// Matrix kernels for regular, wide and wide packed-input tiles of type `ty`.
    #[cfg(not(metal_driver))]
    fn matrix_kernels(&self, ty: TensorType) -> Option<&[Rc<Kernel>; 3]> {
        self.mm
            .iter()
            .find(|(t, _)| *t == ty)
            .map(|(_, kernels)| kernels)
    }

    /// Records a product on the matrix units with the kernel's output tile.
    ///
    /// # Safety
    ///
    /// As [`Self::record_gemv`].
    #[cfg(not(metal_driver))]
    #[allow(clippy::too_many_arguments)]
    unsafe fn record_matrix(
        &self,
        kernel: &Rc<Kernel>,
        w: &DeviceMatrix,
        input: [&Rc<Allocation>; 3],
        y: &DeviceBuffer<f32>,
        out: Out,
        capacity: usize,
        tile: (usize, usize),
        batch: &Batch,
        commands: &mut Commands,
    ) {
        let block = block_len(w.ty).expect("matrix kernels take types with GEMV kernels");
        // SAFETY: matches mm_*.wgsl, which takes gemv.wgsl's bindings and
        // push constants; it reads the rows and writes the outputs of the
        // tokens the batch selects, which the caller vouches for, and
        // tiles past them return early.
        unsafe {
            commands.dispatch(
                kernel,
                &[
                    w.data.allocation(),
                    input[0],
                    input[1],
                    input[2],
                    y.allocation(),
                    batch.device().allocation(),
                ],
                &[
                    w.rows as u32,
                    (w.cols / block) as u32,
                    out.stride as u32,
                    out.offset as u32,
                    0,
                ],
                [
                    w.rows.div_ceil(tile.0) as u32,
                    capacity.max(1).div_ceil(tile.1) as u32,
                    1,
                ],
            )
        };
    }

    /// The kernel for weights of type `ty` and the rows one of its
    /// workgroups takes; with `many`, the one written for a pass of many
    /// tokens, where the driver has it.
    fn gemv_kernel(&self, ty: TensorType, many: bool) -> (&Rc<Kernel>, usize) {
        #[cfg(metal_driver)]
        if many {
            match ty {
                TensorType::Q4K => return (&self.gemv_q4_k_many, MANY_ROWS_PER_GROUP),
                TensorType::Q5K => return (&self.gemv_q5_k_many, MANY_ROWS_PER_GROUP),
                TensorType::Q6K => return (&self.gemv_q6_k_many, MANY_ROWS_PER_GROUP),
                _ => {}
            }
        }
        #[cfg(not(metal_driver))]
        let _ = many;
        let kernel = match ty {
            TensorType::F32 => &self.gemv_f32,
            TensorType::F16 => &self.gemv_f16,
            TensorType::Q8_0 => &self.gemv_q8_0,
            TensorType::Q4K => &self.gemv_q4_k,
            TensorType::Q5K => &self.gemv_q5_k,
            TensorType::Q6K => &self.gemv_q6_k,
            _ => unreachable!("block_len admits only types with kernels"),
        };
        (kernel, ROWS_PER_GROUP)
    }
}
