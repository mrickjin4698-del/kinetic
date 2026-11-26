// First physical block of this row and the distance between its blocks.
// Q8 on Vulkan stores each column of blocks across a group of output rows.
fn weight_row(row: u32, rows: u32, blocks: u32) -> vec2<u32> {
    if WEIGHT_ROWS == 1u {
        return vec2<u32>(row * blocks, 1u);
    }
    let first_row = row / WEIGHT_ROWS * WEIGHT_ROWS;
    let in_group = row - first_row;
    return vec2<u32>(first_row * blocks + in_group, min(WEIGHT_ROWS, rows - first_row));
}
