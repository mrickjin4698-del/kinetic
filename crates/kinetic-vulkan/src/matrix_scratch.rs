use std::rc::Rc;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::memory::DeviceBuffer;

/// Scratch for a prompt's f16 input blocks and one inverse scale per token.
pub struct MatrixScratch {
    pub(crate) values: DeviceBuffer<u32>,
    pub(crate) scales: DeviceBuffer<f32>,
}

impl MatrixScratch {
    /// Holds inputs of up to `width` values and `rows` tokens.
    pub fn new(device: &Rc<Device>, width: usize, rows: usize) -> Result<Self> {
        let elements = width
            .checked_mul(rows)
            .filter(|&n| n > 0 && u32::try_from(n).is_ok() && width.is_multiple_of(32))
            .ok_or_else(|| Error::Shape(format!("matrix scratch of {rows} rows of {width}")))?;
        Ok(Self {
            values: DeviceBuffer::new(device, elements / 2)?,
            scales: DeviceBuffer::new(device, 2 * rows)?,
        })
    }

    /// Device bytes held.
    pub fn size_bytes(&self) -> usize {
        4 * (self.values.len() + self.scales.len())
    }
}
