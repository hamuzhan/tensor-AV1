/*
 * Copyright(c) 2024 Alliance for Open Media
 *
 * Data Collection Pipeline for ML Training
 * Captures encoder decisions (MVs, SADs, partition maps) during encoding
 * and writes them to HDF5 format for training Vision-Mamba-2 models.
 *
 * Enabled via -DENABLE_DATA_COLLECTION=ON at build time.
 */

#ifndef SVT_DATA_COLLECTOR_H
#define SVT_DATA_COLLECTOR_H

#ifdef ENABLE_DATA_COLLECTION

#include "definitions.h"
#include "mv.h"
#include "svt_threads.h"

// Forward declaration (defined in hdf5_writer.c)
struct HDF5WriterState;

#ifdef __cplusplus
extern "C" {
#endif

// ---------- Constants ----------
#define DC_MAX_REF_LISTS       2    // MAX_NUM_OF_REF_PIC_LIST
#define DC_MAX_REFS_PER_LIST   4    // MAX_REF_IDX
#define DC_SQUARE_PU_COUNT     85   // SQUARE_PU_COUNT from me_sb_results.h
#define DC_MAX_PARTITION_DEPTH 6    // 128x128 -> 4x4 = 5 splits + root
#define DC_PARTITION_MAP_DIM   32   // 128/4 = 32 (4x4 resolution grid per SB)

// Validation bounds (derived from AV1 spec enums in definitions.h)
#define DC_MAX_PARTITION_TYPE  9    // PARTITION_VERT_4
#define DC_MAX_BLOCK_SIZE      21   // BLOCK_SIZES_ALL - 1
#define DC_MAX_PRED_MODE       24   // MB_MODE_COUNT - 1
#define DC_INTER_MODE_START    13   // NEARESTMV
#define DC_MV_LIMIT            16384 // MV_UPP = 1 << 14
#define DC_MAX_QP              63
#define DC_MAX_SLICE_TYPE      1    // I_SLICE
#define DC_MAX_TEMPORAL_LAYER  5    // MAX_TEMPORAL_LAYERS - 1
#define DC_MAX_TX_TYPE         16   // TX_TYPES
#define DC_MAX_TX_DEPTH        2    // max transform subdivision depth
#define DC_MAX_REF_FRAME       7    // ALTREF_FRAME

// ---------- Per-SB Motion Estimation Data ----------
typedef struct DcMeData {
    Mv       best_mv[DC_MAX_REF_LISTS][DC_MAX_REFS_PER_LIST][DC_SQUARE_PU_COUNT];
    uint32_t best_sad[DC_MAX_REF_LISTS][DC_MAX_REFS_PER_LIST][DC_SQUARE_PU_COUNT];
    uint16_t sb_origin_x;
    uint16_t sb_origin_y;
    uint16_t sb_width;
    uint16_t sb_height;
    uint8_t  num_refs[DC_MAX_REF_LISTS];
    uint8_t  valid;
} DcMeData;

// ---------- Per-SB Partition / Mode Decision Data ----------
typedef struct DcPartitionData {
    // Each cell represents one 4x4 block within the SB
    uint8_t partition_map[DC_PARTITION_MAP_DIM][DC_PARTITION_MAP_DIM];   // PartitionType
    uint8_t block_size_map[DC_PARTITION_MAP_DIM][DC_PARTITION_MAP_DIM];  // BlockSize enum
    uint8_t pred_mode_map[DC_PARTITION_MAP_DIM][DC_PARTITION_MAP_DIM];   // PredictionMode
    uint8_t is_inter_map[DC_PARTITION_MAP_DIM][DC_PARTITION_MAP_DIM];
    Mv      final_mv[DC_PARTITION_MAP_DIM][DC_PARTITION_MAP_DIM];
    uint8_t tx_type_map[DC_PARTITION_MAP_DIM][DC_PARTITION_MAP_DIM];    // TxType enum
    uint8_t tx_depth_map[DC_PARTITION_MAP_DIM][DC_PARTITION_MAP_DIM];   // 0-2
    int8_t  ref_frame0_map[DC_PARTITION_MAP_DIM][DC_PARTITION_MAP_DIM]; // primary ref
    int8_t  ref_frame1_map[DC_PARTITION_MAP_DIM][DC_PARTITION_MAP_DIM]; // compound ref
    int64_t rd_cost_by_depth[DC_MAX_PARTITION_DEPTH];
    uint16_t sb_origin_x;
    uint16_t sb_origin_y;
    uint8_t  valid;
} DcPartitionData;

// ---------- Per-Frame Metadata ----------
typedef struct DcFrameMetadata {
    uint64_t picture_number;
    uint64_t decode_order;
    uint32_t cur_order_hint;
    uint8_t  slice_type;           // I_SLICE=1, B_SLICE=0
    uint8_t  temporal_layer_index;
    uint8_t  is_ref;
    uint8_t  idr_flag;
    uint8_t  cra_flag;
    uint8_t  scene_change_flag;
    uint8_t  hierarchical_levels;
    uint8_t  qp;
    uint16_t frame_width;
    uint16_t frame_height;
    uint8_t  bit_depth;
    uint8_t  ref_list0_count;
    uint8_t  ref_list1_count;
    uint64_t ref_pic_poc[DC_MAX_REF_LISTS][DC_MAX_REFS_PER_LIST];
} DcFrameMetadata;

// ---------- Per-Frame Collector ----------
typedef struct FrameDataCollector {
    DcFrameMetadata  metadata;
    DcMeData*        me_data;          // [b64_total_count]
    DcPartitionData* partition_data;   // [sb_total_count]

    // Raw frame luma (non-owning pointer into encoder buffer)
    uint8_t* raw_luma_copy;    // owned copy for async writing
    uint16_t raw_luma_stride;
    uint32_t raw_luma_width;
    uint32_t raw_luma_height;

    // Completion tracking
    uint32_t me_sbs_total;
    uint32_t enc_sbs_total;
    EbHandle completion_mutex;
    uint8_t  me_complete;
    uint8_t  enc_complete;
    uint8_t  metadata_set;
    uint8_t  ready_for_write;
    uint8_t  in_use;

    uint64_t picture_number;   // which picture this collector is for
} FrameDataCollector;

// ---------- Global Data Collection Context ----------
typedef struct DataCollectionContext {
    // Configuration
    char     output_path[512];
    uint8_t  capture_raw_frames;
    uint8_t  capture_me;
    uint8_t  capture_partition;
    uint32_t max_frames;           // 0 = unlimited

    // Frame collector pool (ring buffer)
    FrameDataCollector* collectors;
    uint32_t            pool_size;
    EbHandle            pool_mutex;

    // Writer queue
    FrameDataCollector** write_queue;
    uint32_t             write_queue_head;
    uint32_t             write_queue_tail;
    uint32_t             write_queue_capacity;
    EbHandle             write_queue_mutex;
    EbHandle             write_queue_semaphore;

    // Writer thread
    EbHandle writer_thread_handle;
    uint8_t  writer_thread_exit;

    // HDF5 writer state (cached handles, buffers, dataspaces)
    struct HDF5WriterState* hdf5_writer;

    // Stats
    uint64_t frames_written;
    uint64_t frames_dropped;
    uint64_t frames_validated;
    uint64_t validation_failures;

    // Encoder geometry (set once at init)
    uint16_t pic_width;
    uint16_t pic_height;
    uint8_t  bit_depth;
    uint16_t b64_total_count;
    uint16_t sb_total_count;
    uint8_t  sb_size;    // 64 or 128
} DataCollectionContext;

// ---------- Lifecycle API ----------
EbErrorType dc_init(DataCollectionContext** ctx_out,
                    const char* output_path,
                    uint16_t pic_width,
                    uint16_t pic_height,
                    uint8_t bit_depth,
                    uint16_t b64_total_count,
                    uint16_t sb_total_count,
                    uint8_t sb_size,
                    uint32_t pool_size);

void dc_destroy(DataCollectionContext* ctx);

// ---------- Hook API ----------
FrameDataCollector* dc_get_collector(DataCollectionContext* ctx, uint64_t picture_number);

void dc_record_frame_metadata(FrameDataCollector* collector,
                              const DcFrameMetadata* metadata);

void dc_record_me_sb(FrameDataCollector* collector,
                     uint32_t sb_index,
                     const DcMeData* me_data);

void dc_record_partition_sb(FrameDataCollector* collector,
                            uint32_t sb_index,
                            const DcPartitionData* partition_data);

void dc_signal_me_complete(DataCollectionContext* ctx,
                           FrameDataCollector* collector);

void dc_signal_enc_complete(DataCollectionContext* ctx,
                            FrameDataCollector* collector);

#ifdef __cplusplus
}
#endif

#endif // ENABLE_DATA_COLLECTION
#endif // SVT_DATA_COLLECTOR_H
