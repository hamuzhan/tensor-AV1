/*
 * HDF5 serialization backend for the data collection pipeline.
 * Creates flat extensible datasets optimized for ML training consumption.
 *
 * Uses cached dataset handles, compound types, memory dataspaces,
 * and pre-allocated reshape buffers to minimize per-frame overhead.
 */

#ifndef SVT_HDF5_WRITER_H
#define SVT_HDF5_WRITER_H

#ifdef ENABLE_DATA_COLLECTION

#include "data_collector.h"

// Opaque writer state -- definition in hdf5_writer.c
typedef struct HDF5WriterState HDF5WriterState;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize HDF5 writer state (creates file, datasets, caches handles and buffers)
// Returns allocated state pointer, or NULL on error
HDF5WriterState* hdf5_writer_init(const char* output_path,
                                   uint16_t pic_width,
                                   uint16_t pic_height,
                                   uint8_t bit_depth,
                                   uint16_t b64_total_count,
                                   uint16_t sb_total_count);

// Append one frame's data using cached handles and pre-allocated buffers
// Returns 0 on success, negative on error
int hdf5_writer_append_frame(HDF5WriterState* state,
                              const FrameDataCollector* collector,
                              uint16_t b64_total_count,
                              uint16_t sb_total_count);

// Write final sequence-level stats as HDF5 file attributes before closing
void hdf5_writer_set_final_stats(HDF5WriterState* state,
                                  uint64_t total_frames,
                                  uint64_t validation_failures);

// Close all cached handles, free buffers, close file, free state
void hdf5_writer_close(HDF5WriterState* state);

#ifdef __cplusplus
}
#endif

#endif // ENABLE_DATA_COLLECTION
#endif // SVT_HDF5_WRITER_H
