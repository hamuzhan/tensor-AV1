/*
 * HDF5 serialization backend for the data collection pipeline.
 * Creates flat extensible datasets optimized for ML training consumption.
 */

#ifndef SVT_HDF5_WRITER_H
#define SVT_HDF5_WRITER_H

#ifdef ENABLE_DATA_COLLECTION

#include "data_collector.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize HDF5 file and create all datasets
// Returns HDF5 file handle (hid_t) cast to int64_t, or negative on error
int64_t hdf5_writer_init(const char* output_path,
                         uint16_t pic_width,
                         uint16_t pic_height,
                         uint8_t bit_depth,
                         uint16_t b64_total_count,
                         uint16_t sb_total_count);

// Append one frame's data to the HDF5 file
// Returns 0 on success, negative on error
int hdf5_writer_append_frame(int64_t file_handle,
                             const FrameDataCollector* collector,
                             uint16_t b64_total_count,
                             uint16_t sb_total_count);

// Close the HDF5 file
void hdf5_writer_close(int64_t file_handle);

#ifdef __cplusplus
}
#endif

#endif // ENABLE_DATA_COLLECTION
#endif // SVT_HDF5_WRITER_H
