/*
 * HDF5 serialization backend.
 * Creates flat extensible datasets for ML training consumption.
 *
 * Dataset layout:
 *   /metadata          compound[N]
 *   /me/mvs            int16[N, b64s, 2, 4, 85, 2]
 *   /me/sads           uint32[N, b64s, 2, 4, 85]
 *   /partition/map     uint8[N, sbs, 32, 32]
 *   /partition/bsize   uint8[N, sbs, 32, 32]
 *   /partition/pmode   uint8[N, sbs, 32, 32]
 *   /partition/inter   uint8[N, sbs, 32, 32]
 *   /partition/fmv     int16[N, sbs, 32, 32, 2]
 *   /partition/rdcost  int64[N, sbs, 6]
 *   /frames/luma       uint8[N, H, W]  (optional)
 */

#ifdef ENABLE_DATA_COLLECTION

#include "hdf5_writer.h"
#include "svt_log.h"

#include <hdf5.h>
#include <string.h>
#include <stdlib.h>

#define DC_CHUNK_FRAMES 1  // one frame per chunk (optimal for sequential append)

// Helper: create an extensible dataset with given dimensions
static hid_t create_extensible_dataset(hid_t parent, const char* name,
                                       hid_t dtype, int rank,
                                       const hsize_t* dims,
                                       const hsize_t* chunk_dims) {
    hsize_t maxdims[8];
    for (int i = 0; i < rank; i++)
        maxdims[i] = dims[i];
    maxdims[0] = H5S_UNLIMITED;  // first dim is extensible

    hid_t space = H5Screate_simple(rank, dims, maxdims);
    if (space < 0)
        return -1;

    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist, rank, chunk_dims);
    H5Pset_deflate(plist, 1);  // light gzip compression

    hid_t dset = H5Dcreate2(parent, name, dtype, space, H5P_DEFAULT, plist, H5P_DEFAULT);
    H5Pclose(plist);
    H5Sclose(space);
    return dset;
}

int64_t hdf5_writer_init(const char* output_path,
                         uint16_t pic_width,
                         uint16_t pic_height,
                         uint8_t bit_depth,
                         uint16_t b64_total_count,
                         uint16_t sb_total_count) {
    hid_t file = H5Fcreate(output_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0)
        return -1;

    // Store encoder geometry as file attributes
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr;

    attr = H5Acreate2(file, "pic_width", H5T_NATIVE_UINT16, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT16, &pic_width);
    H5Aclose(attr);

    attr = H5Acreate2(file, "pic_height", H5T_NATIVE_UINT16, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT16, &pic_height);
    H5Aclose(attr);

    attr = H5Acreate2(file, "bit_depth", H5T_NATIVE_UINT8, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT8, &bit_depth);
    H5Aclose(attr);

    attr = H5Acreate2(file, "b64_total_count", H5T_NATIVE_UINT16, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT16, &b64_total_count);
    H5Aclose(attr);

    attr = H5Acreate2(file, "sb_total_count", H5T_NATIVE_UINT16, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT16, &sb_total_count);
    H5Aclose(attr);

    H5Sclose(attr_space);

    // Create groups
    hid_t me_group = H5Gcreate2(file, "/me", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t part_group = H5Gcreate2(file, "/partition", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t frames_group = H5Gcreate2(file, "/frames", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Create metadata compound type
    hid_t meta_type = H5Tcreate(H5T_COMPOUND, sizeof(DcFrameMetadata));
    H5Tinsert(meta_type, "picture_number", HOFFSET(DcFrameMetadata, picture_number), H5T_NATIVE_UINT64);
    H5Tinsert(meta_type, "decode_order", HOFFSET(DcFrameMetadata, decode_order), H5T_NATIVE_UINT64);
    H5Tinsert(meta_type, "cur_order_hint", HOFFSET(DcFrameMetadata, cur_order_hint), H5T_NATIVE_UINT32);
    H5Tinsert(meta_type, "slice_type", HOFFSET(DcFrameMetadata, slice_type), H5T_NATIVE_UINT8);
    H5Tinsert(meta_type, "temporal_layer_index", HOFFSET(DcFrameMetadata, temporal_layer_index), H5T_NATIVE_UINT8);
    H5Tinsert(meta_type, "is_ref", HOFFSET(DcFrameMetadata, is_ref), H5T_NATIVE_UINT8);
    H5Tinsert(meta_type, "idr_flag", HOFFSET(DcFrameMetadata, idr_flag), H5T_NATIVE_UINT8);
    H5Tinsert(meta_type, "cra_flag", HOFFSET(DcFrameMetadata, cra_flag), H5T_NATIVE_UINT8);
    H5Tinsert(meta_type, "scene_change_flag", HOFFSET(DcFrameMetadata, scene_change_flag), H5T_NATIVE_UINT8);
    H5Tinsert(meta_type, "hierarchical_levels", HOFFSET(DcFrameMetadata, hierarchical_levels), H5T_NATIVE_UINT8);
    H5Tinsert(meta_type, "qp", HOFFSET(DcFrameMetadata, qp), H5T_NATIVE_UINT8);
    H5Tinsert(meta_type, "frame_width", HOFFSET(DcFrameMetadata, frame_width), H5T_NATIVE_UINT16);
    H5Tinsert(meta_type, "frame_height", HOFFSET(DcFrameMetadata, frame_height), H5T_NATIVE_UINT16);
    H5Tinsert(meta_type, "bit_depth", HOFFSET(DcFrameMetadata, bit_depth), H5T_NATIVE_UINT8);
    H5Tinsert(meta_type, "ref_list0_count", HOFFSET(DcFrameMetadata, ref_list0_count), H5T_NATIVE_UINT8);
    H5Tinsert(meta_type, "ref_list1_count", HOFFSET(DcFrameMetadata, ref_list1_count), H5T_NATIVE_UINT8);

    // Metadata dataset: [0] initially, extensible
    hsize_t meta_dims[1] = {0};
    hsize_t meta_chunk[1] = {DC_CHUNK_FRAMES};
    hid_t meta_dset = create_extensible_dataset(file, "/metadata", meta_type, 1,
                                                 meta_dims, meta_chunk);

    // ME MVs: [0, b64s, 2, 4, 85, 2] int16
    {
        hsize_t dims[6] = {0, b64_total_count, DC_MAX_REF_LISTS, DC_MAX_REFS_PER_LIST, DC_SQUARE_PU_COUNT, 2};
        hsize_t chunk[6] = {DC_CHUNK_FRAMES, b64_total_count, DC_MAX_REF_LISTS, DC_MAX_REFS_PER_LIST, DC_SQUARE_PU_COUNT, 2};
        create_extensible_dataset(me_group, "mvs", H5T_NATIVE_INT16, 6, dims, chunk);
    }

    // ME SADs: [0, b64s, 2, 4, 85] uint32
    {
        hsize_t dims[5] = {0, b64_total_count, DC_MAX_REF_LISTS, DC_MAX_REFS_PER_LIST, DC_SQUARE_PU_COUNT};
        hsize_t chunk[5] = {DC_CHUNK_FRAMES, b64_total_count, DC_MAX_REF_LISTS, DC_MAX_REFS_PER_LIST, DC_SQUARE_PU_COUNT};
        create_extensible_dataset(me_group, "sads", H5T_NATIVE_UINT32, 5, dims, chunk);
    }

    // Partition datasets: [0, sbs, 32, 32]
    {
        hsize_t dims[4] = {0, sb_total_count, DC_PARTITION_MAP_DIM, DC_PARTITION_MAP_DIM};
        hsize_t chunk[4] = {DC_CHUNK_FRAMES, sb_total_count, DC_PARTITION_MAP_DIM, DC_PARTITION_MAP_DIM};
        create_extensible_dataset(part_group, "map", H5T_NATIVE_UINT8, 4, dims, chunk);
        create_extensible_dataset(part_group, "bsize", H5T_NATIVE_UINT8, 4, dims, chunk);
        create_extensible_dataset(part_group, "pmode", H5T_NATIVE_UINT8, 4, dims, chunk);
        create_extensible_dataset(part_group, "inter", H5T_NATIVE_UINT8, 4, dims, chunk);
    }

    // Partition final MVs: [0, sbs, 32, 32, 2]
    {
        hsize_t dims[5] = {0, sb_total_count, DC_PARTITION_MAP_DIM, DC_PARTITION_MAP_DIM, 2};
        hsize_t chunk[5] = {DC_CHUNK_FRAMES, sb_total_count, DC_PARTITION_MAP_DIM, DC_PARTITION_MAP_DIM, 2};
        create_extensible_dataset(part_group, "fmv", H5T_NATIVE_INT16, 5, dims, chunk);
    }

    // Partition RD costs: [0, sbs, 6]
    {
        hsize_t dims[3] = {0, sb_total_count, DC_MAX_PARTITION_DEPTH};
        hsize_t chunk[3] = {DC_CHUNK_FRAMES, sb_total_count, DC_MAX_PARTITION_DEPTH};
        create_extensible_dataset(part_group, "rdcost", H5T_NATIVE_INT64, 3, dims, chunk);
    }

    // Optional luma: [0, H, W]
    {
        hsize_t dims[3] = {0, pic_height, pic_width};
        hsize_t chunk[3] = {DC_CHUNK_FRAMES, pic_height, pic_width};
        create_extensible_dataset(frames_group, "luma", H5T_NATIVE_UINT8, 3, dims, chunk);
    }

    H5Tclose(meta_type);
    H5Gclose(me_group);
    H5Gclose(part_group);
    H5Gclose(frames_group);

    return (int64_t)file;
}

// Helper: extend dataset by 1 along dim 0 and write a hyperslab
static int append_to_dataset(hid_t file, const char* path,
                             const void* data, hid_t mem_type,
                             int rank, const hsize_t* frame_dims) {
    hid_t dset = H5Dopen2(file, path, H5P_DEFAULT);
    if (dset < 0)
        return -1;

    // Get current size
    hid_t space = H5Dget_space(dset);
    hsize_t cur_dims[8];
    H5Sget_simple_extent_dims(space, cur_dims, NULL);
    H5Sclose(space);

    // Extend by 1 frame
    hsize_t new_dims[8];
    for (int i = 0; i < rank; i++)
        new_dims[i] = (i == 0) ? cur_dims[0] + 1 : cur_dims[i];
    H5Dset_extent(dset, new_dims);

    // Select hyperslab for the new frame
    hid_t file_space = H5Dget_space(dset);
    hsize_t offset[8];
    hsize_t count[8];
    for (int i = 0; i < rank; i++) {
        offset[i] = (i == 0) ? cur_dims[0] : 0;
        count[i]  = frame_dims[i];
    }
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, NULL, count, NULL);

    // Memory space
    hid_t mem_space = H5Screate_simple(rank, frame_dims, NULL);

    herr_t status = H5Dwrite(dset, mem_type, mem_space, file_space, H5P_DEFAULT, data);

    H5Sclose(mem_space);
    H5Sclose(file_space);
    H5Dclose(dset);

    return (status < 0) ? -1 : 0;
}

int hdf5_writer_append_frame(int64_t file_handle,
                             const FrameDataCollector* fc,
                             uint16_t b64_total_count,
                             uint16_t sb_total_count) {
    hid_t file = (hid_t)file_handle;
    if (file < 0 || !fc)
        return -1;

    // 1. Write metadata
    {
        // Recreate compound type for writing (must match creation)
        hid_t meta_type = H5Tcreate(H5T_COMPOUND, sizeof(DcFrameMetadata));
        H5Tinsert(meta_type, "picture_number", HOFFSET(DcFrameMetadata, picture_number), H5T_NATIVE_UINT64);
        H5Tinsert(meta_type, "decode_order", HOFFSET(DcFrameMetadata, decode_order), H5T_NATIVE_UINT64);
        H5Tinsert(meta_type, "cur_order_hint", HOFFSET(DcFrameMetadata, cur_order_hint), H5T_NATIVE_UINT32);
        H5Tinsert(meta_type, "slice_type", HOFFSET(DcFrameMetadata, slice_type), H5T_NATIVE_UINT8);
        H5Tinsert(meta_type, "temporal_layer_index", HOFFSET(DcFrameMetadata, temporal_layer_index), H5T_NATIVE_UINT8);
        H5Tinsert(meta_type, "is_ref", HOFFSET(DcFrameMetadata, is_ref), H5T_NATIVE_UINT8);
        H5Tinsert(meta_type, "idr_flag", HOFFSET(DcFrameMetadata, idr_flag), H5T_NATIVE_UINT8);
        H5Tinsert(meta_type, "cra_flag", HOFFSET(DcFrameMetadata, cra_flag), H5T_NATIVE_UINT8);
        H5Tinsert(meta_type, "scene_change_flag", HOFFSET(DcFrameMetadata, scene_change_flag), H5T_NATIVE_UINT8);
        H5Tinsert(meta_type, "hierarchical_levels", HOFFSET(DcFrameMetadata, hierarchical_levels), H5T_NATIVE_UINT8);
        H5Tinsert(meta_type, "qp", HOFFSET(DcFrameMetadata, qp), H5T_NATIVE_UINT8);
        H5Tinsert(meta_type, "frame_width", HOFFSET(DcFrameMetadata, frame_width), H5T_NATIVE_UINT16);
        H5Tinsert(meta_type, "frame_height", HOFFSET(DcFrameMetadata, frame_height), H5T_NATIVE_UINT16);
        H5Tinsert(meta_type, "bit_depth", HOFFSET(DcFrameMetadata, bit_depth), H5T_NATIVE_UINT8);
        H5Tinsert(meta_type, "ref_list0_count", HOFFSET(DcFrameMetadata, ref_list0_count), H5T_NATIVE_UINT8);
        H5Tinsert(meta_type, "ref_list1_count", HOFFSET(DcFrameMetadata, ref_list1_count), H5T_NATIVE_UINT8);

        hsize_t frame_dims[1] = {1};
        append_to_dataset(file, "/metadata", &fc->metadata, meta_type, 1, frame_dims);
        H5Tclose(meta_type);
    }

    // 2. Write ME MVs - reshape from DcMeData array to contiguous buffer
    {
        size_t mv_count = (size_t)b64_total_count * DC_MAX_REF_LISTS * DC_MAX_REFS_PER_LIST * DC_SQUARE_PU_COUNT * 2;
        int16_t* mv_buf = (int16_t*)malloc(mv_count * sizeof(int16_t));
        if (mv_buf) {
            size_t idx = 0;
            for (uint16_t sb = 0; sb < b64_total_count; sb++) {
                for (int list = 0; list < DC_MAX_REF_LISTS; list++) {
                    for (int ref = 0; ref < DC_MAX_REFS_PER_LIST; ref++) {
                        for (int pu = 0; pu < DC_SQUARE_PU_COUNT; pu++) {
                            mv_buf[idx++] = fc->me_data[sb].best_mv[list][ref][pu].x;
                            mv_buf[idx++] = fc->me_data[sb].best_mv[list][ref][pu].y;
                        }
                    }
                }
            }
            hsize_t frame_dims[6] = {1, b64_total_count, DC_MAX_REF_LISTS, DC_MAX_REFS_PER_LIST, DC_SQUARE_PU_COUNT, 2};
            append_to_dataset(file, "/me/mvs", mv_buf, H5T_NATIVE_INT16, 6, frame_dims);
            free(mv_buf);
        }
    }

    // 3. Write ME SADs
    {
        size_t sad_count = (size_t)b64_total_count * DC_MAX_REF_LISTS * DC_MAX_REFS_PER_LIST * DC_SQUARE_PU_COUNT;
        uint32_t* sad_buf = (uint32_t*)malloc(sad_count * sizeof(uint32_t));
        if (sad_buf) {
            size_t idx = 0;
            for (uint16_t sb = 0; sb < b64_total_count; sb++) {
                for (int list = 0; list < DC_MAX_REF_LISTS; list++) {
                    for (int ref = 0; ref < DC_MAX_REFS_PER_LIST; ref++) {
                        for (int pu = 0; pu < DC_SQUARE_PU_COUNT; pu++) {
                            sad_buf[idx++] = fc->me_data[sb].best_sad[list][ref][pu];
                        }
                    }
                }
            }
            hsize_t frame_dims[5] = {1, b64_total_count, DC_MAX_REF_LISTS, DC_MAX_REFS_PER_LIST, DC_SQUARE_PU_COUNT};
            append_to_dataset(file, "/me/sads", sad_buf, H5T_NATIVE_UINT32, 5, frame_dims);
            free(sad_buf);
        }
    }

    // 4. Write partition maps (4 maps + fmv + rdcost)
    {
        size_t map_count = (size_t)sb_total_count * DC_PARTITION_MAP_DIM * DC_PARTITION_MAP_DIM;
        uint8_t* map_buf   = (uint8_t*)malloc(map_count);
        uint8_t* bsize_buf = (uint8_t*)malloc(map_count);
        uint8_t* pmode_buf = (uint8_t*)malloc(map_count);
        uint8_t* inter_buf = (uint8_t*)malloc(map_count);

        if (map_buf && bsize_buf && pmode_buf && inter_buf) {
            size_t idx = 0;
            for (uint16_t sb = 0; sb < sb_total_count; sb++) {
                for (int r = 0; r < DC_PARTITION_MAP_DIM; r++) {
                    for (int c = 0; c < DC_PARTITION_MAP_DIM; c++) {
                        map_buf[idx]   = fc->partition_data[sb].partition_map[r][c];
                        bsize_buf[idx] = fc->partition_data[sb].block_size_map[r][c];
                        pmode_buf[idx] = fc->partition_data[sb].pred_mode_map[r][c];
                        inter_buf[idx] = fc->partition_data[sb].is_inter_map[r][c];
                        idx++;
                    }
                }
            }
            hsize_t frame_dims[4] = {1, sb_total_count, DC_PARTITION_MAP_DIM, DC_PARTITION_MAP_DIM};
            append_to_dataset(file, "/partition/map", map_buf, H5T_NATIVE_UINT8, 4, frame_dims);
            append_to_dataset(file, "/partition/bsize", bsize_buf, H5T_NATIVE_UINT8, 4, frame_dims);
            append_to_dataset(file, "/partition/pmode", pmode_buf, H5T_NATIVE_UINT8, 4, frame_dims);
            append_to_dataset(file, "/partition/inter", inter_buf, H5T_NATIVE_UINT8, 4, frame_dims);
        }
        free(map_buf);
        free(bsize_buf);
        free(pmode_buf);
        free(inter_buf);

        // Final MVs
        size_t fmv_count = (size_t)sb_total_count * DC_PARTITION_MAP_DIM * DC_PARTITION_MAP_DIM * 2;
        int16_t* fmv_buf = (int16_t*)malloc(fmv_count * sizeof(int16_t));
        if (fmv_buf) {
            size_t idx = 0;
            for (uint16_t sb = 0; sb < sb_total_count; sb++) {
                for (int r = 0; r < DC_PARTITION_MAP_DIM; r++) {
                    for (int c = 0; c < DC_PARTITION_MAP_DIM; c++) {
                        fmv_buf[idx++] = fc->partition_data[sb].final_mv[r][c].x;
                        fmv_buf[idx++] = fc->partition_data[sb].final_mv[r][c].y;
                    }
                }
            }
            hsize_t frame_dims[5] = {1, sb_total_count, DC_PARTITION_MAP_DIM, DC_PARTITION_MAP_DIM, 2};
            append_to_dataset(file, "/partition/fmv", fmv_buf, H5T_NATIVE_INT16, 5, frame_dims);
            free(fmv_buf);
        }

        // RD costs
        size_t rd_count = (size_t)sb_total_count * DC_MAX_PARTITION_DEPTH;
        int64_t* rd_buf = (int64_t*)malloc(rd_count * sizeof(int64_t));
        if (rd_buf) {
            size_t idx = 0;
            for (uint16_t sb = 0; sb < sb_total_count; sb++) {
                for (int d = 0; d < DC_MAX_PARTITION_DEPTH; d++) {
                    rd_buf[idx++] = fc->partition_data[sb].rd_cost_by_depth[d];
                }
            }
            hsize_t frame_dims[3] = {1, sb_total_count, DC_MAX_PARTITION_DEPTH};
            append_to_dataset(file, "/partition/rdcost", rd_buf, H5T_NATIVE_INT64, 3, frame_dims);
            free(rd_buf);
        }
    }

    // 5. Write raw luma (if available)
    if (fc->raw_luma_copy && fc->raw_luma_width > 0 && fc->raw_luma_height > 0) {
        hsize_t frame_dims[3] = {1, fc->raw_luma_height, fc->raw_luma_width};
        append_to_dataset(file, "/frames/luma", fc->raw_luma_copy, H5T_NATIVE_UINT8, 3, frame_dims);
    }

    return 0;
}

void hdf5_writer_close(int64_t file_handle) {
    hid_t file = (hid_t)file_handle;
    if (file >= 0) {
        H5Fflush(file, H5F_SCOPE_GLOBAL);
        H5Fclose(file);
    }
}

#endif // ENABLE_DATA_COLLECTION
