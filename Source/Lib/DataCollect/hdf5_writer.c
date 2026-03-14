/*
 * HDF5 serialization backend.
 * Creates flat extensible datasets for ML training consumption.
 *
 * Performance: Dataset handles, compound types, memory dataspaces,
 * and reshape buffers are cached at init and reused every frame.
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

// ---------- Dataset and memory-space index enums ----------

typedef enum {
    DSET_METADATA = 0,
    DSET_ME_MVS,
    DSET_ME_SADS,
    DSET_PART_MAP,
    DSET_PART_BSIZE,
    DSET_PART_PMODE,
    DSET_PART_INTER,
    DSET_PART_FMV,
    DSET_PART_RDCOST,
    DSET_FRAMES_LUMA,
    DSET_COUNT  // = 10
} DatasetIndex;

typedef enum {
    MSPACE_META = 0,       // rank 1: {1}
    MSPACE_ME_MVS,         // rank 6: {1, b64, 2, 4, 85, 2}
    MSPACE_ME_SADS,        // rank 5: {1, b64, 2, 4, 85}
    MSPACE_PART_MAP,       // rank 4: {1, sb, 32, 32} — shared by 4 partition datasets
    MSPACE_PART_FMV,       // rank 5: {1, sb, 32, 32, 2}
    MSPACE_PART_RDCOST,    // rank 3: {1, sb, 6}
    MSPACE_FRAMES_LUMA,    // rank 3: {1, H, W}
    MSPACE_COUNT  // = 7
} MemSpaceIndex;

// ---------- Writer state (cached across all frame writes) ----------

struct HDF5WriterState {
    hid_t    file;
    hid_t    datasets[DSET_COUNT];
    hid_t    meta_type;
    hid_t    mem_spaces[MSPACE_COUNT];
    int      ranks[DSET_COUNT];
    hsize_t  frame_dims[DSET_COUNT][8];
    MemSpaceIndex dset_to_mspace[DSET_COUNT];
    hsize_t  frame_count;

    // Pre-allocated reshape buffers
    int16_t*  mv_buf;
    uint32_t* sad_buf;
    uint8_t*  map_buf;
    uint8_t*  bsize_buf;
    uint8_t*  pmode_buf;
    uint8_t*  inter_buf;
    int16_t*  fmv_buf;
    int64_t*  rd_buf;
};

// ---------- Helpers ----------

// Create an extensible dataset with given dimensions
static hid_t create_extensible_dataset(hid_t parent, const char* name,
                                       hid_t dtype, int rank,
                                       const hsize_t* dims,
                                       const hsize_t* chunk_dims) {
    hsize_t maxdims[8];
    for (int i = 0; i < rank; i++)
        maxdims[i] = dims[i];
    maxdims[0] = H5S_UNLIMITED;

    hid_t space = H5Screate_simple(rank, dims, maxdims);
    if (space < 0)
        return -1;

    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist, rank, chunk_dims);
    H5Pset_deflate(plist, 1);

    hid_t dset = H5Dcreate2(parent, name, dtype, space, H5P_DEFAULT, plist, H5P_DEFAULT);
    H5Pclose(plist);
    H5Sclose(space);
    return dset;
}

// Extend a cached dataset by 1 frame and write a hyperslab
static int append_to_cached_dataset(HDF5WriterState* state,
                                     DatasetIndex dset_idx,
                                     const void* data,
                                     hid_t mem_type) {
    hid_t dset = state->datasets[dset_idx];
    if (dset < 0)
        return -1;

    int rank = state->ranks[dset_idx];
    const hsize_t* fdims = state->frame_dims[dset_idx];

    // Extend — use tracked frame_count instead of H5Dget_space
    hsize_t new_dims[8];
    new_dims[0] = state->frame_count + 1;
    for (int i = 1; i < rank; i++)
        new_dims[i] = fdims[i];
    H5Dset_extent(dset, new_dims);

    // Select hyperslab for the new frame
    hid_t file_space = H5Dget_space(dset);
    hsize_t offset[8];
    hsize_t count[8];
    for (int i = 0; i < rank; i++) {
        offset[i] = (i == 0) ? state->frame_count : 0;
        count[i]  = fdims[i];
    }
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, NULL, count, NULL);

    // Use cached memory dataspace
    hid_t mem_space = state->mem_spaces[state->dset_to_mspace[dset_idx]];

    herr_t status = H5Dwrite(dset, mem_type, mem_space, file_space, H5P_DEFAULT, data);
    H5Sclose(file_space);

    return (status < 0) ? -1 : 0;
}

// ---------- Lifecycle ----------

HDF5WriterState* hdf5_writer_init(const char* output_path,
                                   uint16_t pic_width,
                                   uint16_t pic_height,
                                   uint8_t bit_depth,
                                   uint16_t b64_total_count,
                                   uint16_t sb_total_count) {
    HDF5WriterState* state = (HDF5WriterState*)calloc(1, sizeof(HDF5WriterState));
    if (!state)
        return NULL;

    // Init all hid_t to -1 for safe partial cleanup
    state->file = -1;
    state->meta_type = -1;
    for (int i = 0; i < DSET_COUNT; i++)
        state->datasets[i] = -1;
    for (int i = 0; i < MSPACE_COUNT; i++)
        state->mem_spaces[i] = -1;
    state->frame_count = 0;

    // Create HDF5 file
    state->file = H5Fcreate(output_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (state->file < 0) {
        hdf5_writer_close(state);
        return NULL;
    }

    // Store encoder geometry as file attributes
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr;

    attr = H5Acreate2(state->file, "pic_width", H5T_NATIVE_UINT16, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT16, &pic_width);
    H5Aclose(attr);

    attr = H5Acreate2(state->file, "pic_height", H5T_NATIVE_UINT16, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT16, &pic_height);
    H5Aclose(attr);

    attr = H5Acreate2(state->file, "bit_depth", H5T_NATIVE_UINT8, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT8, &bit_depth);
    H5Aclose(attr);

    attr = H5Acreate2(state->file, "b64_total_count", H5T_NATIVE_UINT16, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT16, &b64_total_count);
    H5Aclose(attr);

    attr = H5Acreate2(state->file, "sb_total_count", H5T_NATIVE_UINT16, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT16, &sb_total_count);
    H5Aclose(attr);

    H5Sclose(attr_space);

    // Create groups
    hid_t me_group = H5Gcreate2(state->file, "/me", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t part_group = H5Gcreate2(state->file, "/partition", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t frames_group = H5Gcreate2(state->file, "/frames", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Build metadata compound type (cached — NOT closed until hdf5_writer_close)
    state->meta_type = H5Tcreate(H5T_COMPOUND, sizeof(DcFrameMetadata));
    H5Tinsert(state->meta_type, "picture_number", HOFFSET(DcFrameMetadata, picture_number), H5T_NATIVE_UINT64);
    H5Tinsert(state->meta_type, "decode_order", HOFFSET(DcFrameMetadata, decode_order), H5T_NATIVE_UINT64);
    H5Tinsert(state->meta_type, "cur_order_hint", HOFFSET(DcFrameMetadata, cur_order_hint), H5T_NATIVE_UINT32);
    H5Tinsert(state->meta_type, "slice_type", HOFFSET(DcFrameMetadata, slice_type), H5T_NATIVE_UINT8);
    H5Tinsert(state->meta_type, "temporal_layer_index", HOFFSET(DcFrameMetadata, temporal_layer_index), H5T_NATIVE_UINT8);
    H5Tinsert(state->meta_type, "is_ref", HOFFSET(DcFrameMetadata, is_ref), H5T_NATIVE_UINT8);
    H5Tinsert(state->meta_type, "idr_flag", HOFFSET(DcFrameMetadata, idr_flag), H5T_NATIVE_UINT8);
    H5Tinsert(state->meta_type, "cra_flag", HOFFSET(DcFrameMetadata, cra_flag), H5T_NATIVE_UINT8);
    H5Tinsert(state->meta_type, "scene_change_flag", HOFFSET(DcFrameMetadata, scene_change_flag), H5T_NATIVE_UINT8);
    H5Tinsert(state->meta_type, "hierarchical_levels", HOFFSET(DcFrameMetadata, hierarchical_levels), H5T_NATIVE_UINT8);
    H5Tinsert(state->meta_type, "qp", HOFFSET(DcFrameMetadata, qp), H5T_NATIVE_UINT8);
    H5Tinsert(state->meta_type, "frame_width", HOFFSET(DcFrameMetadata, frame_width), H5T_NATIVE_UINT16);
    H5Tinsert(state->meta_type, "frame_height", HOFFSET(DcFrameMetadata, frame_height), H5T_NATIVE_UINT16);
    H5Tinsert(state->meta_type, "bit_depth", HOFFSET(DcFrameMetadata, bit_depth), H5T_NATIVE_UINT8);
    H5Tinsert(state->meta_type, "ref_list0_count", HOFFSET(DcFrameMetadata, ref_list0_count), H5T_NATIVE_UINT8);
    H5Tinsert(state->meta_type, "ref_list1_count", HOFFSET(DcFrameMetadata, ref_list1_count), H5T_NATIVE_UINT8);

    // ref_pic_poc: 2D array [DC_MAX_REF_LISTS][DC_MAX_REFS_PER_LIST] of uint64
    {
        hsize_t poc_dims[2] = {DC_MAX_REF_LISTS, DC_MAX_REFS_PER_LIST};
        hid_t poc_array_type = H5Tarray_create2(H5T_NATIVE_UINT64, 2, poc_dims);
        H5Tinsert(state->meta_type, "ref_pic_poc",
                  HOFFSET(DcFrameMetadata, ref_pic_poc), poc_array_type);
        H5Tclose(poc_array_type);
    }

    // Create datasets and cache handles (NOT closed until hdf5_writer_close)
    // Metadata: [0] extensible
    {
        hsize_t dims[1] = {0};
        hsize_t chunk[1] = {DC_CHUNK_FRAMES};
        state->datasets[DSET_METADATA] = create_extensible_dataset(
            state->file, "/metadata", state->meta_type, 1, dims, chunk);
        state->ranks[DSET_METADATA] = 1;
        state->frame_dims[DSET_METADATA][0] = 1;
    }

    // ME MVs: [0, b64s, 2, 4, 85, 2]
    {
        hsize_t dims[6] = {0, b64_total_count, DC_MAX_REF_LISTS, DC_MAX_REFS_PER_LIST, DC_SQUARE_PU_COUNT, 2};
        hsize_t chunk[6] = {DC_CHUNK_FRAMES, b64_total_count, DC_MAX_REF_LISTS, DC_MAX_REFS_PER_LIST, DC_SQUARE_PU_COUNT, 2};
        state->datasets[DSET_ME_MVS] = create_extensible_dataset(
            me_group, "mvs", H5T_NATIVE_INT16, 6, dims, chunk);
        state->ranks[DSET_ME_MVS] = 6;
        state->frame_dims[DSET_ME_MVS][0] = 1;
        state->frame_dims[DSET_ME_MVS][1] = b64_total_count;
        state->frame_dims[DSET_ME_MVS][2] = DC_MAX_REF_LISTS;
        state->frame_dims[DSET_ME_MVS][3] = DC_MAX_REFS_PER_LIST;
        state->frame_dims[DSET_ME_MVS][4] = DC_SQUARE_PU_COUNT;
        state->frame_dims[DSET_ME_MVS][5] = 2;
    }

    // ME SADs: [0, b64s, 2, 4, 85]
    {
        hsize_t dims[5] = {0, b64_total_count, DC_MAX_REF_LISTS, DC_MAX_REFS_PER_LIST, DC_SQUARE_PU_COUNT};
        hsize_t chunk[5] = {DC_CHUNK_FRAMES, b64_total_count, DC_MAX_REF_LISTS, DC_MAX_REFS_PER_LIST, DC_SQUARE_PU_COUNT};
        state->datasets[DSET_ME_SADS] = create_extensible_dataset(
            me_group, "sads", H5T_NATIVE_UINT32, 5, dims, chunk);
        state->ranks[DSET_ME_SADS] = 5;
        state->frame_dims[DSET_ME_SADS][0] = 1;
        state->frame_dims[DSET_ME_SADS][1] = b64_total_count;
        state->frame_dims[DSET_ME_SADS][2] = DC_MAX_REF_LISTS;
        state->frame_dims[DSET_ME_SADS][3] = DC_MAX_REFS_PER_LIST;
        state->frame_dims[DSET_ME_SADS][4] = DC_SQUARE_PU_COUNT;
    }

    // Partition datasets: [0, sbs, 32, 32]
    {
        hsize_t dims[4] = {0, sb_total_count, DC_PARTITION_MAP_DIM, DC_PARTITION_MAP_DIM};
        hsize_t chunk[4] = {DC_CHUNK_FRAMES, sb_total_count, DC_PARTITION_MAP_DIM, DC_PARTITION_MAP_DIM};
        state->datasets[DSET_PART_MAP] = create_extensible_dataset(
            part_group, "map", H5T_NATIVE_UINT8, 4, dims, chunk);
        state->datasets[DSET_PART_BSIZE] = create_extensible_dataset(
            part_group, "bsize", H5T_NATIVE_UINT8, 4, dims, chunk);
        state->datasets[DSET_PART_PMODE] = create_extensible_dataset(
            part_group, "pmode", H5T_NATIVE_UINT8, 4, dims, chunk);
        state->datasets[DSET_PART_INTER] = create_extensible_dataset(
            part_group, "inter", H5T_NATIVE_UINT8, 4, dims, chunk);
        for (int i = DSET_PART_MAP; i <= DSET_PART_INTER; i++) {
            state->ranks[i] = 4;
            state->frame_dims[i][0] = 1;
            state->frame_dims[i][1] = sb_total_count;
            state->frame_dims[i][2] = DC_PARTITION_MAP_DIM;
            state->frame_dims[i][3] = DC_PARTITION_MAP_DIM;
        }
    }

    // Partition final MVs: [0, sbs, 32, 32, 2]
    {
        hsize_t dims[5] = {0, sb_total_count, DC_PARTITION_MAP_DIM, DC_PARTITION_MAP_DIM, 2};
        hsize_t chunk[5] = {DC_CHUNK_FRAMES, sb_total_count, DC_PARTITION_MAP_DIM, DC_PARTITION_MAP_DIM, 2};
        state->datasets[DSET_PART_FMV] = create_extensible_dataset(
            part_group, "fmv", H5T_NATIVE_INT16, 5, dims, chunk);
        state->ranks[DSET_PART_FMV] = 5;
        state->frame_dims[DSET_PART_FMV][0] = 1;
        state->frame_dims[DSET_PART_FMV][1] = sb_total_count;
        state->frame_dims[DSET_PART_FMV][2] = DC_PARTITION_MAP_DIM;
        state->frame_dims[DSET_PART_FMV][3] = DC_PARTITION_MAP_DIM;
        state->frame_dims[DSET_PART_FMV][4] = 2;
    }

    // Partition RD costs: [0, sbs, 6]
    {
        hsize_t dims[3] = {0, sb_total_count, DC_MAX_PARTITION_DEPTH};
        hsize_t chunk[3] = {DC_CHUNK_FRAMES, sb_total_count, DC_MAX_PARTITION_DEPTH};
        state->datasets[DSET_PART_RDCOST] = create_extensible_dataset(
            part_group, "rdcost", H5T_NATIVE_INT64, 3, dims, chunk);
        state->ranks[DSET_PART_RDCOST] = 3;
        state->frame_dims[DSET_PART_RDCOST][0] = 1;
        state->frame_dims[DSET_PART_RDCOST][1] = sb_total_count;
        state->frame_dims[DSET_PART_RDCOST][2] = DC_MAX_PARTITION_DEPTH;
    }

    // Optional luma: [0, H, W]
    {
        hsize_t dims[3] = {0, pic_height, pic_width};
        hsize_t chunk[3] = {DC_CHUNK_FRAMES, pic_height, pic_width};
        state->datasets[DSET_FRAMES_LUMA] = create_extensible_dataset(
            frames_group, "luma", H5T_NATIVE_UINT8, 3, dims, chunk);
        state->ranks[DSET_FRAMES_LUMA] = 3;
        state->frame_dims[DSET_FRAMES_LUMA][0] = 1;
        state->frame_dims[DSET_FRAMES_LUMA][1] = pic_height;
        state->frame_dims[DSET_FRAMES_LUMA][2] = pic_width;
    }

    // Close groups (not needed after dataset creation — datasets are accessed via cached handles)
    H5Gclose(me_group);
    H5Gclose(part_group);
    H5Gclose(frames_group);

    // Verify all datasets were created
    for (int i = 0; i < DSET_COUNT; i++) {
        if (state->datasets[i] < 0) {
            SVT_ERROR("DC: Failed to create dataset index %d\n", i);
            hdf5_writer_close(state);
            return NULL;
        }
    }

    // Create cached memory dataspaces (7 distinct shapes, reused every frame)
    {
        hsize_t meta_fdims[1] = {1};
        state->mem_spaces[MSPACE_META] = H5Screate_simple(1, meta_fdims, NULL);
    }
    state->mem_spaces[MSPACE_ME_MVS] = H5Screate_simple(
        6, state->frame_dims[DSET_ME_MVS], NULL);
    state->mem_spaces[MSPACE_ME_SADS] = H5Screate_simple(
        5, state->frame_dims[DSET_ME_SADS], NULL);
    state->mem_spaces[MSPACE_PART_MAP] = H5Screate_simple(
        4, state->frame_dims[DSET_PART_MAP], NULL);
    state->mem_spaces[MSPACE_PART_FMV] = H5Screate_simple(
        5, state->frame_dims[DSET_PART_FMV], NULL);
    state->mem_spaces[MSPACE_PART_RDCOST] = H5Screate_simple(
        3, state->frame_dims[DSET_PART_RDCOST], NULL);
    state->mem_spaces[MSPACE_FRAMES_LUMA] = H5Screate_simple(
        3, state->frame_dims[DSET_FRAMES_LUMA], NULL);

    for (int i = 0; i < MSPACE_COUNT; i++) {
        if (state->mem_spaces[i] < 0) {
            SVT_ERROR("DC: Failed to create memory dataspace index %d\n", i);
            hdf5_writer_close(state);
            return NULL;
        }
    }

    // Dataset-to-memory-space mapping
    state->dset_to_mspace[DSET_METADATA]    = MSPACE_META;
    state->dset_to_mspace[DSET_ME_MVS]      = MSPACE_ME_MVS;
    state->dset_to_mspace[DSET_ME_SADS]     = MSPACE_ME_SADS;
    state->dset_to_mspace[DSET_PART_MAP]    = MSPACE_PART_MAP;
    state->dset_to_mspace[DSET_PART_BSIZE]  = MSPACE_PART_MAP;  // shared
    state->dset_to_mspace[DSET_PART_PMODE]  = MSPACE_PART_MAP;  // shared
    state->dset_to_mspace[DSET_PART_INTER]  = MSPACE_PART_MAP;  // shared
    state->dset_to_mspace[DSET_PART_FMV]    = MSPACE_PART_FMV;
    state->dset_to_mspace[DSET_PART_RDCOST] = MSPACE_PART_RDCOST;
    state->dset_to_mspace[DSET_FRAMES_LUMA] = MSPACE_FRAMES_LUMA;

    // Pre-allocate reshape buffers (allocated once, reused every frame)
    size_t mv_count  = (size_t)b64_total_count * DC_MAX_REF_LISTS * DC_MAX_REFS_PER_LIST * DC_SQUARE_PU_COUNT * 2;
    size_t sad_count = (size_t)b64_total_count * DC_MAX_REF_LISTS * DC_MAX_REFS_PER_LIST * DC_SQUARE_PU_COUNT;
    size_t map_count = (size_t)sb_total_count * DC_PARTITION_MAP_DIM * DC_PARTITION_MAP_DIM;
    size_t fmv_count = (size_t)sb_total_count * DC_PARTITION_MAP_DIM * DC_PARTITION_MAP_DIM * 2;
    size_t rd_count  = (size_t)sb_total_count * DC_MAX_PARTITION_DEPTH;

    state->mv_buf    = (int16_t*)malloc(mv_count * sizeof(int16_t));
    state->sad_buf   = (uint32_t*)malloc(sad_count * sizeof(uint32_t));
    state->map_buf   = (uint8_t*)malloc(map_count);
    state->bsize_buf = (uint8_t*)malloc(map_count);
    state->pmode_buf = (uint8_t*)malloc(map_count);
    state->inter_buf = (uint8_t*)malloc(map_count);
    state->fmv_buf   = (int16_t*)malloc(fmv_count * sizeof(int16_t));
    state->rd_buf    = (int64_t*)malloc(rd_count * sizeof(int64_t));

    if (!state->mv_buf || !state->sad_buf || !state->map_buf ||
        !state->bsize_buf || !state->pmode_buf || !state->inter_buf ||
        !state->fmv_buf || !state->rd_buf) {
        SVT_ERROR("DC: Failed to allocate reshape buffers\n");
        hdf5_writer_close(state);
        return NULL;
    }

    return state;
}

// ---------- Per-frame append ----------

int hdf5_writer_append_frame(HDF5WriterState* state,
                              const FrameDataCollector* fc,
                              uint16_t b64_total_count,
                              uint16_t sb_total_count) {
    if (!state || state->file < 0 || !fc)
        return -1;

    int errors = 0;

    // 1. Write metadata (uses cached compound type)
    errors += (append_to_cached_dataset(state, DSET_METADATA, &fc->metadata, state->meta_type) < 0);

    // 2. Write ME MVs — reshape from DcMeData array to contiguous buffer
    {
        size_t idx = 0;
        for (uint16_t sb = 0; sb < b64_total_count; sb++) {
            for (int list = 0; list < DC_MAX_REF_LISTS; list++) {
                for (int ref = 0; ref < DC_MAX_REFS_PER_LIST; ref++) {
                    for (int pu = 0; pu < DC_SQUARE_PU_COUNT; pu++) {
                        state->mv_buf[idx++] = fc->me_data[sb].best_mv[list][ref][pu].x;
                        state->mv_buf[idx++] = fc->me_data[sb].best_mv[list][ref][pu].y;
                    }
                }
            }
        }
        errors += (append_to_cached_dataset(state, DSET_ME_MVS, state->mv_buf, H5T_NATIVE_INT16) < 0);
    }

    // 3. Write ME SADs
    {
        size_t idx = 0;
        for (uint16_t sb = 0; sb < b64_total_count; sb++) {
            for (int list = 0; list < DC_MAX_REF_LISTS; list++) {
                for (int ref = 0; ref < DC_MAX_REFS_PER_LIST; ref++) {
                    for (int pu = 0; pu < DC_SQUARE_PU_COUNT; pu++) {
                        state->sad_buf[idx++] = fc->me_data[sb].best_sad[list][ref][pu];
                    }
                }
            }
        }
        errors += (append_to_cached_dataset(state, DSET_ME_SADS, state->sad_buf, H5T_NATIVE_UINT32) < 0);
    }

    // 4. Write partition maps (4 maps + fmv + rdcost)
    {
        size_t idx = 0;
        for (uint16_t sb = 0; sb < sb_total_count; sb++) {
            for (int r = 0; r < DC_PARTITION_MAP_DIM; r++) {
                for (int c = 0; c < DC_PARTITION_MAP_DIM; c++) {
                    state->map_buf[idx]   = fc->partition_data[sb].partition_map[r][c];
                    state->bsize_buf[idx] = fc->partition_data[sb].block_size_map[r][c];
                    state->pmode_buf[idx] = fc->partition_data[sb].pred_mode_map[r][c];
                    state->inter_buf[idx] = fc->partition_data[sb].is_inter_map[r][c];
                    idx++;
                }
            }
        }
        errors += (append_to_cached_dataset(state, DSET_PART_MAP,   state->map_buf,   H5T_NATIVE_UINT8) < 0);
        errors += (append_to_cached_dataset(state, DSET_PART_BSIZE, state->bsize_buf, H5T_NATIVE_UINT8) < 0);
        errors += (append_to_cached_dataset(state, DSET_PART_PMODE, state->pmode_buf, H5T_NATIVE_UINT8) < 0);
        errors += (append_to_cached_dataset(state, DSET_PART_INTER, state->inter_buf, H5T_NATIVE_UINT8) < 0);

        // Final MVs
        idx = 0;
        for (uint16_t sb = 0; sb < sb_total_count; sb++) {
            for (int r = 0; r < DC_PARTITION_MAP_DIM; r++) {
                for (int c = 0; c < DC_PARTITION_MAP_DIM; c++) {
                    state->fmv_buf[idx++] = fc->partition_data[sb].final_mv[r][c].x;
                    state->fmv_buf[idx++] = fc->partition_data[sb].final_mv[r][c].y;
                }
            }
        }
        errors += (append_to_cached_dataset(state, DSET_PART_FMV, state->fmv_buf, H5T_NATIVE_INT16) < 0);

        // RD costs
        idx = 0;
        for (uint16_t sb = 0; sb < sb_total_count; sb++) {
            for (int d = 0; d < DC_MAX_PARTITION_DEPTH; d++) {
                state->rd_buf[idx++] = fc->partition_data[sb].rd_cost_by_depth[d];
            }
        }
        errors += (append_to_cached_dataset(state, DSET_PART_RDCOST, state->rd_buf, H5T_NATIVE_INT64) < 0);
    }

    // 5. Write raw luma (if available)
    if (fc->raw_luma_copy && fc->raw_luma_width > 0 && fc->raw_luma_height > 0) {
        errors += (append_to_cached_dataset(state, DSET_FRAMES_LUMA, fc->raw_luma_copy, H5T_NATIVE_UINT8) < 0);
    }

    if (errors > 0)
        return -1;

    state->frame_count++;
    return 0;
}

// ---------- Final Stats ----------

void hdf5_writer_set_final_stats(HDF5WriterState* state,
                                  uint64_t total_frames,
                                  uint64_t validation_failures) {
    if (!state || state->file < 0)
        return;

    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr;

    attr = H5Acreate2(state->file, "total_frames_written", H5T_NATIVE_UINT64,
                      attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT64, &total_frames);
    H5Aclose(attr);

    attr = H5Acreate2(state->file, "validation_failures", H5T_NATIVE_UINT64,
                      attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_UINT64, &validation_failures);
    H5Aclose(attr);

    H5Sclose(attr_space);
}

// ---------- Cleanup ----------

void hdf5_writer_close(HDF5WriterState* state) {
    if (!state)
        return;

    // Close cached dataset handles
    for (int i = 0; i < DSET_COUNT; i++) {
        if (state->datasets[i] >= 0)
            H5Dclose(state->datasets[i]);
    }

    // Close cached metadata compound type
    if (state->meta_type >= 0)
        H5Tclose(state->meta_type);

    // Close cached memory dataspaces
    for (int i = 0; i < MSPACE_COUNT; i++) {
        if (state->mem_spaces[i] >= 0)
            H5Sclose(state->mem_spaces[i]);
    }

    // Flush and close the file
    if (state->file >= 0) {
        H5Fflush(state->file, H5F_SCOPE_GLOBAL);
        H5Fclose(state->file);
    }

    // Free pre-allocated buffers
    free(state->mv_buf);
    free(state->sad_buf);
    free(state->map_buf);
    free(state->bsize_buf);
    free(state->pmode_buf);
    free(state->inter_buf);
    free(state->fmv_buf);
    free(state->rd_buf);

    free(state);
}

#endif // ENABLE_DATA_COLLECTION
