/*
 * Recursively walks a PC_TREE and flattens partition decisions
 * into a 32x32 grid at 4x4 block resolution.
 */

#ifdef ENABLE_DATA_COLLECTION

#include "partition_tree_flatten.h"
#include "md_process.h"
#include "common_utils.h"
#include "coding_unit.h"

#include <string.h>

// Extracted block info passed between extract and fill functions
typedef struct {
    uint8_t pred_mode;
    uint8_t is_inter;
    Mv      mv;
    uint8_t tx_type;
    uint8_t tx_depth;
    int8_t  ref_frame0;
    int8_t  ref_frame1;
} BlockInfo;

// Fill a rectangular region of the partition map with values from a leaf block
static void fill_block_region(DcPartitionData* out,
                              int mi_row, int mi_col,
                              int mi_row_start, int mi_col_start,
                              int bw_mi, int bh_mi,
                              uint8_t partition_type,
                              uint8_t block_size,
                              const BlockInfo* bi) {
    for (int r = 0; r < bh_mi; r++) {
        for (int c = 0; c < bw_mi; c++) {
            int grid_r = (mi_row - mi_row_start) + r;
            int grid_c = (mi_col - mi_col_start) + c;
            if (grid_r >= 0 && grid_r < DC_PARTITION_MAP_DIM &&
                grid_c >= 0 && grid_c < DC_PARTITION_MAP_DIM) {
                out->partition_map[grid_r][grid_c]  = partition_type;
                out->block_size_map[grid_r][grid_c] = block_size;
                out->pred_mode_map[grid_r][grid_c]  = bi->pred_mode;
                out->is_inter_map[grid_r][grid_c]   = bi->is_inter;
                out->final_mv[grid_r][grid_c]       = bi->mv;
                out->tx_type_map[grid_r][grid_c]    = bi->tx_type;
                out->tx_depth_map[grid_r][grid_c]   = bi->tx_depth;
                out->ref_frame0_map[grid_r][grid_c] = bi->ref_frame0;
                out->ref_frame1_map[grid_r][grid_c] = bi->ref_frame1;
            }
        }
    }
}

// Extract mode info from a BlkStruct at a given shape/index in PC_TREE
static void extract_block_info(const PC_TREE* pc_tree, int shape, int nsi,
                               BlockInfo* bi) {
    bi->pred_mode  = 0;
    bi->is_inter   = 0;
    bi->mv.as_int  = 0;
    bi->tx_type    = 0;
    bi->tx_depth   = 0;
    bi->ref_frame0 = -1;
    bi->ref_frame1 = -1;

    if (pc_tree->block_data[shape][nsi]) {
        const BlkStruct* blk = pc_tree->block_data[shape][nsi];
        bi->pred_mode  = (uint8_t)blk->block_mi.mode;
        bi->is_inter   = (blk->block_mi.mode >= NEARESTMV) ? 1 : 0;
        bi->tx_type    = (uint8_t)blk->tx_type[0];
        bi->tx_depth   = blk->block_mi.tx_depth;
        bi->ref_frame0 = (int8_t)blk->block_mi.ref_frame[0];
        bi->ref_frame1 = (int8_t)blk->block_mi.ref_frame[1];
        if (bi->is_inter)
            bi->mv = blk->block_mi.mv[0];
    }
}

// Recursive flattener
static void flatten_recursive(const PC_TREE* pc_tree,
                              DcPartitionData* out,
                              int mi_row, int mi_col,
                              int mi_row_start, int mi_col_start,
                              int depth) {
    if (!pc_tree || depth >= DC_MAX_PARTITION_DEPTH)
        return;

    BlockSize bsize = pc_tree->bsize;
    int bw = block_size_wide[bsize];
    int bh = block_size_high[bsize];
    int bw_mi = bw >> 2;  // width in 4x4 MI units
    int bh_mi = bh >> 2;

    // Record RD cost at this depth
    if (depth < DC_MAX_PARTITION_DEPTH && pc_tree->rdc.valid)
        out->rd_cost_by_depth[depth] = pc_tree->rdc.rd_cost;

    PartitionType partition = pc_tree->partition;
    BlockInfo bi;

    if (partition == PARTITION_NONE) {
        extract_block_info(pc_tree, PART_N, 0, &bi);
        fill_block_region(out, mi_row, mi_col, mi_row_start, mi_col_start,
                          bw_mi, bh_mi,
                          (uint8_t)PARTITION_NONE, (uint8_t)bsize, &bi);
    } else if (partition == PARTITION_SPLIT) {
        int half_w = bw_mi >> 1;
        int half_h = bh_mi >> 1;

        if (pc_tree->split[0])
            flatten_recursive(pc_tree->split[0], out,
                              mi_row, mi_col,
                              mi_row_start, mi_col_start, depth + 1);
        if (pc_tree->split[1])
            flatten_recursive(pc_tree->split[1], out,
                              mi_row, mi_col + half_w,
                              mi_row_start, mi_col_start, depth + 1);
        if (pc_tree->split[2])
            flatten_recursive(pc_tree->split[2], out,
                              mi_row + half_h, mi_col,
                              mi_row_start, mi_col_start, depth + 1);
        if (pc_tree->split[3])
            flatten_recursive(pc_tree->split[3], out,
                              mi_row + half_h, mi_col + half_w,
                              mi_row_start, mi_col_start, depth + 1);
    } else if (partition == PARTITION_HORZ) {
        int half_h = bh_mi >> 1;

        extract_block_info(pc_tree, PART_H, 0, &bi);
        fill_block_region(out, mi_row, mi_col, mi_row_start, mi_col_start,
                          bw_mi, half_h,
                          (uint8_t)PARTITION_HORZ, (uint8_t)bsize, &bi);

        extract_block_info(pc_tree, PART_H, 1, &bi);
        fill_block_region(out, mi_row + half_h, mi_col, mi_row_start, mi_col_start,
                          bw_mi, half_h,
                          (uint8_t)PARTITION_HORZ, (uint8_t)bsize, &bi);
    } else if (partition == PARTITION_VERT) {
        int half_w = bw_mi >> 1;

        extract_block_info(pc_tree, PART_V, 0, &bi);
        fill_block_region(out, mi_row, mi_col, mi_row_start, mi_col_start,
                          half_w, bh_mi,
                          (uint8_t)PARTITION_VERT, (uint8_t)bsize, &bi);

        extract_block_info(pc_tree, PART_V, 1, &bi);
        fill_block_region(out, mi_row, mi_col + half_w, mi_row_start, mi_col_start,
                          half_w, bh_mi,
                          (uint8_t)PARTITION_VERT, (uint8_t)bsize, &bi);
    } else if (partition == PARTITION_HORZ_A) {
        int half_w = bw_mi >> 1;
        int half_h = bh_mi >> 1;

        extract_block_info(pc_tree, PART_HA, 0, &bi);
        fill_block_region(out, mi_row, mi_col, mi_row_start, mi_col_start,
                          half_w, half_h,
                          (uint8_t)partition, (uint8_t)bsize, &bi);

        extract_block_info(pc_tree, PART_HA, 1, &bi);
        fill_block_region(out, mi_row, mi_col + half_w, mi_row_start, mi_col_start,
                          half_w, half_h,
                          (uint8_t)partition, (uint8_t)bsize, &bi);

        extract_block_info(pc_tree, PART_HA, 2, &bi);
        fill_block_region(out, mi_row + half_h, mi_col, mi_row_start, mi_col_start,
                          bw_mi, half_h,
                          (uint8_t)partition, (uint8_t)bsize, &bi);
    } else if (partition == PARTITION_HORZ_B) {
        int half_w = bw_mi >> 1;
        int half_h = bh_mi >> 1;

        extract_block_info(pc_tree, PART_HB, 0, &bi);
        fill_block_region(out, mi_row, mi_col, mi_row_start, mi_col_start,
                          bw_mi, half_h,
                          (uint8_t)partition, (uint8_t)bsize, &bi);

        extract_block_info(pc_tree, PART_HB, 1, &bi);
        fill_block_region(out, mi_row + half_h, mi_col, mi_row_start, mi_col_start,
                          half_w, half_h,
                          (uint8_t)partition, (uint8_t)bsize, &bi);

        extract_block_info(pc_tree, PART_HB, 2, &bi);
        fill_block_region(out, mi_row + half_h, mi_col + half_w, mi_row_start, mi_col_start,
                          half_w, half_h,
                          (uint8_t)partition, (uint8_t)bsize, &bi);
    } else if (partition == PARTITION_VERT_A) {
        int half_w = bw_mi >> 1;
        int half_h = bh_mi >> 1;

        extract_block_info(pc_tree, PART_VA, 0, &bi);
        fill_block_region(out, mi_row, mi_col, mi_row_start, mi_col_start,
                          half_w, half_h,
                          (uint8_t)partition, (uint8_t)bsize, &bi);

        extract_block_info(pc_tree, PART_VA, 1, &bi);
        fill_block_region(out, mi_row + half_h, mi_col, mi_row_start, mi_col_start,
                          half_w, half_h,
                          (uint8_t)partition, (uint8_t)bsize, &bi);

        extract_block_info(pc_tree, PART_VA, 2, &bi);
        fill_block_region(out, mi_row, mi_col + half_w, mi_row_start, mi_col_start,
                          half_w, bh_mi,
                          (uint8_t)partition, (uint8_t)bsize, &bi);
    } else if (partition == PARTITION_VERT_B) {
        int half_w = bw_mi >> 1;
        int half_h = bh_mi >> 1;

        extract_block_info(pc_tree, PART_VB, 0, &bi);
        fill_block_region(out, mi_row, mi_col, mi_row_start, mi_col_start,
                          half_w, bh_mi,
                          (uint8_t)partition, (uint8_t)bsize, &bi);

        extract_block_info(pc_tree, PART_VB, 1, &bi);
        fill_block_region(out, mi_row, mi_col + half_w, mi_row_start, mi_col_start,
                          half_w, half_h,
                          (uint8_t)partition, (uint8_t)bsize, &bi);

        extract_block_info(pc_tree, PART_VB, 2, &bi);
        fill_block_region(out, mi_row + half_h, mi_col + half_w, mi_row_start, mi_col_start,
                          half_w, half_h,
                          (uint8_t)partition, (uint8_t)bsize, &bi);
    } else if (partition == PARTITION_HORZ_4) {
        int quarter_h = bh_mi >> 2;
        for (int i = 0; i < 4; i++) {
            extract_block_info(pc_tree, PART_H4, i, &bi);
            fill_block_region(out, mi_row + i * quarter_h, mi_col,
                              mi_row_start, mi_col_start,
                              bw_mi, quarter_h,
                              (uint8_t)partition, (uint8_t)bsize, &bi);
        }
    } else if (partition == PARTITION_VERT_4) {
        int quarter_w = bw_mi >> 2;
        for (int i = 0; i < 4; i++) {
            extract_block_info(pc_tree, PART_V4, i, &bi);
            fill_block_region(out, mi_row, mi_col + i * quarter_w,
                              mi_row_start, mi_col_start,
                              quarter_w, bh_mi,
                              (uint8_t)partition, (uint8_t)bsize, &bi);
        }
    }
}

void dc_flatten_partition_tree(const PC_TREE* pc_tree,
                               DcPartitionData* out,
                               int mi_row_start,
                               int mi_col_start,
                               uint8_t sb_size_log2) {
    if (!pc_tree || !out)
        return;
    (void)sb_size_log2;  // pc_tree->bsize carries this info
    memset(out->partition_map, 0, sizeof(out->partition_map));
    memset(out->block_size_map, 0, sizeof(out->block_size_map));
    memset(out->pred_mode_map, 0, sizeof(out->pred_mode_map));
    memset(out->is_inter_map, 0, sizeof(out->is_inter_map));
    memset(out->final_mv, 0, sizeof(out->final_mv));
    memset(out->tx_type_map, 0, sizeof(out->tx_type_map));
    memset(out->tx_depth_map, 0, sizeof(out->tx_depth_map));
    memset(out->ref_frame0_map, 0xFF, sizeof(out->ref_frame0_map));  // -1 = NONE_FRAME
    memset(out->ref_frame1_map, 0xFF, sizeof(out->ref_frame1_map));
    memset(out->rd_cost_by_depth, 0, sizeof(out->rd_cost_by_depth));

    flatten_recursive(pc_tree, out, mi_row_start, mi_col_start,
                      mi_row_start, mi_col_start, 0);

    out->valid = 1;
}

#endif // ENABLE_DATA_COLLECTION
