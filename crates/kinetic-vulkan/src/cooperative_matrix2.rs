use std::ffi::{c_void, CStr};
use std::ptr;

use ash::vk;

use crate::instance::Instance;

pub(crate) const NAME: &CStr = c"VK_NV_cooperative_matrix2";

// Match mm_q8_0_cm2.comp.
pub(crate) const TILE: (usize, usize) = (64, 128);
const K: u32 = 32;
const WORKGROUP_SIZE: u32 = 128;
const SHARED_BYTES: u32 = 16 + TILE.0 as u32 * 12 * 8;

// These types mirror Vulkan-Headers. Ash 0.38 predates the extension.
#[repr(C)]
pub(crate) struct Features {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    workgroup_scope: vk::Bool32,
    flexible_dimensions: vk::Bool32,
    reductions: vk::Bool32,
    conversions: vk::Bool32,
    per_element_operations: vk::Bool32,
    tensor_addressing: vk::Bool32,
    block_loads: vk::Bool32,
}

// SAFETY: matches VkPhysicalDeviceCooperativeMatrix2FeaturesNV. The
// registry allows it in both pNext chains.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for Features {}
unsafe impl vk::ExtendsDeviceCreateInfo for Features {}

impl Features {
    pub(crate) fn new(enable: bool) -> Self {
        let on = vk::Bool32::from(enable);
        Self {
            s_type: vk::StructureType::from_raw(1_000_593_000),
            p_next: ptr::null_mut(),
            workgroup_scope: on,
            flexible_dimensions: on,
            reductions: vk::FALSE,
            conversions: vk::FALSE,
            per_element_operations: on,
            tensor_addressing: on,
            block_loads: vk::FALSE,
        }
    }

    pub(crate) fn supported(&self) -> bool {
        self.workgroup_scope == vk::TRUE
            && self.flexible_dimensions == vk::TRUE
            && self.per_element_operations == vk::TRUE
            && self.tensor_addressing == vk::TRUE
    }
}

#[repr(C)]
pub(crate) struct Properties {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    max_workgroup_size: u32,
    max_dimension: u32,
    reserved_shared_memory: u32,
}

// SAFETY: matches VkPhysicalDeviceCooperativeMatrix2PropertiesNV, which
// extends VkPhysicalDeviceProperties2.
unsafe impl vk::ExtendsPhysicalDeviceProperties2 for Properties {}

impl Default for Properties {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1_000_593_002),
            p_next: ptr::null_mut(),
            max_workgroup_size: 0,
            max_dimension: 0,
            reserved_shared_memory: 0,
        }
    }
}

impl Properties {
    pub(crate) fn supports_tile(&self, shared_memory: u32) -> bool {
        self.max_workgroup_size >= WORKGROUP_SIZE
            && self.max_dimension >= (TILE.0.max(TILE.1) as u32).max(K)
            && shared_memory.saturating_sub(self.reserved_shared_memory) >= SHARED_BYTES
    }
}

#[repr(C)]
struct FlexibleShape {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    m: u32,
    n: u32,
    k: u32,
    a_type: vk::ComponentTypeKHR,
    b_type: vk::ComponentTypeKHR,
    c_type: vk::ComponentTypeKHR,
    result_type: vk::ComponentTypeKHR,
    saturating_accumulation: vk::Bool32,
    scope: vk::ScopeKHR,
    invocations: u32,
}

impl Default for FlexibleShape {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1_000_593_001),
            p_next: ptr::null_mut(),
            m: 0,
            n: 0,
            k: 0,
            a_type: vk::ComponentTypeKHR::default(),
            b_type: vk::ComponentTypeKHR::default(),
            c_type: vk::ComponentTypeKHR::default(),
            result_type: vk::ComponentTypeKHR::default(),
            saturating_accumulation: vk::FALSE,
            scope: vk::ScopeKHR::default(),
            invocations: 0,
        }
    }
}

/// Whether the driver supports the shader's f16 workgroup product into f32.
pub(crate) fn has_tile(instance: &Instance, raw: vk::PhysicalDevice) -> bool {
    type GetShapes =
        unsafe extern "system" fn(vk::PhysicalDevice, *mut u32, *mut FlexibleShape) -> vk::Result;

    // SAFETY: the caller found this extension on a device from the instance.
    let Some(entry) = (unsafe {
        instance.entry.get_instance_proc_addr(
            instance.raw.handle(),
            c"vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV".as_ptr(),
        )
    }) else {
        return false;
    };
    // SAFETY: GetShapes is the signature of the queried Vulkan function.
    let get: GetShapes = unsafe { std::mem::transmute(entry) };
    let mut count = 0;
    // SAFETY: raw belongs to instance, count is writable, and a null output
    // array requests its length.
    if unsafe { get(raw, &mut count, ptr::null_mut()) } != vk::Result::SUCCESS || count == 0 {
        return false;
    }
    let mut shapes: Vec<_> = (0..count).map(|_| FlexibleShape::default()).collect();
    // SAFETY: shapes has count initialized Vulkan output structures.
    if unsafe { get(raw, &mut count, shapes.as_mut_ptr()) } != vk::Result::SUCCESS {
        return false;
    }
    shapes.truncate(count as usize);
    shapes.iter().any(|p| {
        p.scope == vk::ScopeKHR::WORKGROUP
            && p.invocations == WORKGROUP_SIZE
            && p.a_type == vk::ComponentTypeKHR::FLOAT16
            && p.b_type == vk::ComponentTypeKHR::FLOAT16
            && p.c_type == vk::ComponentTypeKHR::FLOAT32
            && p.result_type == vk::ComponentTypeKHR::FLOAT32
            && p.saturating_accumulation == vk::FALSE
            && (TILE.0 as u32).is_multiple_of(p.m)
            && (TILE.1 as u32).is_multiple_of(p.n)
            && K.is_multiple_of(p.k)
    })
}
