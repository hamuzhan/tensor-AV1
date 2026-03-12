/*
 * Utility to recursively walk a PC_TREE and flatten it into
 * a 32x32 grid (4x4 block resolution) for ML training data.
 */

#ifndef SVT_PARTITION_TREE_FLATTEN_H
#define SVT_PARTITION_TREE_FLATTEN_H

#ifdef ENABLE_DATA_COLLECTION

#include "data_collector.h"

// Forward declare PC_TREE (defined in md_process.h)
struct PC_TREE;

#ifdef __cplusplus
extern "C" {
#endif

// Flatten a PC_TREE into a DcPartitionData struct.
// mi_row, mi_col: position of this SB in MI units (4-pixel blocks)
// sb_size_log2: 6 for 64x64 SB, 7 for 128x128 SB
void dc_flatten_partition_tree(const struct PC_TREE* pc_tree,
                               DcPartitionData* out,
                               int mi_row_start,
                               int mi_col_start,
                               uint8_t sb_size_log2);

#ifdef __cplusplus
}
#endif

#endif // ENABLE_DATA_COLLECTION
#endif // SVT_PARTITION_TREE_FLATTEN_H
