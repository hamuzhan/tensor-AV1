/*
 * Data Collection Pipeline - Core Implementation
 *
 * Thread-safe accumulation of encoder decisions into per-frame collectors,
 * with async HDF5 writing on a dedicated thread.
 */

#ifdef ENABLE_DATA_COLLECTION

#include "data_collector.h"
#include "hdf5_writer.h"
#include "svt_log.h"

#include <stdlib.h>
#include <string.h>

// ---------- Writer Thread ----------

static void* dc_writer_thread(void* arg) {
    DataCollectionContext* ctx = (DataCollectionContext*)arg;

    while (1) {
        svt_block_on_semaphore(ctx->write_queue_semaphore);

        if (ctx->writer_thread_exit)
            break;

        // Dequeue a collector
        svt_block_on_mutex(ctx->write_queue_mutex);
        FrameDataCollector* fc = NULL;
        if (ctx->write_queue_head != ctx->write_queue_tail) {
            fc = ctx->write_queue[ctx->write_queue_head];
            ctx->write_queue_head = (ctx->write_queue_head + 1) % ctx->write_queue_capacity;
        }
        svt_release_mutex(ctx->write_queue_mutex);

        if (!fc)
            continue;

        // Write to HDF5
        int ret = hdf5_writer_append_frame(ctx->hdf5_file, fc,
                                           ctx->b64_total_count,
                                           ctx->sb_total_count);
        if (ret == 0) {
            ctx->frames_written++;
        } else {
            ctx->frames_dropped++;
            SVT_WARN("DC: Failed to write frame %llu to HDF5\n",
                     (unsigned long long)fc->picture_number);
        }

        // Free raw luma copy if allocated
        if (fc->raw_luma_copy) {
            free(fc->raw_luma_copy);
            fc->raw_luma_copy = NULL;
        }

        // Release collector
        svt_block_on_mutex(ctx->pool_mutex);
        fc->in_use         = 0;
        fc->ready_for_write = 0;
        fc->me_complete    = 0;
        fc->enc_complete   = 0;
        fc->metadata_set   = 0;
        svt_release_mutex(ctx->pool_mutex);

        // Check frame limit
        if (ctx->max_frames > 0 && ctx->frames_written >= ctx->max_frames)
            break;
    }
    return NULL;
}

// ---------- Lifecycle ----------

EbErrorType dc_init(DataCollectionContext** ctx_out,
                    const char* output_path,
                    uint16_t pic_width,
                    uint16_t pic_height,
                    uint8_t bit_depth,
                    uint16_t b64_total_count,
                    uint16_t sb_total_count,
                    uint8_t sb_size,
                    uint32_t pool_size) {
    DataCollectionContext* ctx = (DataCollectionContext*)calloc(1, sizeof(DataCollectionContext));
    if (!ctx)
        return EB_ErrorInsufficientResources;

    // Configuration
    strncpy(ctx->output_path, output_path, sizeof(ctx->output_path) - 1);
    ctx->capture_raw_frames = 0;
    ctx->capture_me         = 1;
    ctx->capture_partition  = 1;
    ctx->max_frames         = 0;
    ctx->pic_width          = pic_width;
    ctx->pic_height         = pic_height;
    ctx->bit_depth          = bit_depth;
    ctx->b64_total_count    = b64_total_count;
    ctx->sb_total_count     = sb_total_count;
    ctx->sb_size            = sb_size;
    ctx->pool_size          = pool_size;

    // Check env vars for optional config
    const char* env_raw = getenv("SVT_AV1_DC_CAPTURE_RAW");
    if (env_raw && atoi(env_raw) > 0)
        ctx->capture_raw_frames = 1;

    const char* env_max = getenv("SVT_AV1_DC_MAX_FRAMES");
    if (env_max)
        ctx->max_frames = (uint32_t)atoi(env_max);

    // Create mutexes
    ctx->pool_mutex = svt_create_mutex();
    ctx->write_queue_mutex = svt_create_mutex();
    ctx->write_queue_semaphore = svt_create_semaphore(0, pool_size * 2);

    if (!ctx->pool_mutex || !ctx->write_queue_mutex || !ctx->write_queue_semaphore) {
        dc_destroy(ctx);
        return EB_ErrorInsufficientResources;
    }

    // Allocate collector pool
    ctx->collectors = (FrameDataCollector*)calloc(pool_size, sizeof(FrameDataCollector));
    if (!ctx->collectors) {
        dc_destroy(ctx);
        return EB_ErrorInsufficientResources;
    }

    for (uint32_t i = 0; i < pool_size; i++) {
        FrameDataCollector* fc = &ctx->collectors[i];
        fc->completion_mutex = svt_create_mutex();
        if (!fc->completion_mutex) {
            dc_destroy(ctx);
            return EB_ErrorInsufficientResources;
        }

        fc->me_data = (DcMeData*)calloc(b64_total_count, sizeof(DcMeData));
        fc->partition_data = (DcPartitionData*)calloc(sb_total_count, sizeof(DcPartitionData));
        if (!fc->me_data || !fc->partition_data) {
            dc_destroy(ctx);
            return EB_ErrorInsufficientResources;
        }

        fc->me_sbs_total  = b64_total_count;
        fc->enc_sbs_total = sb_total_count;
    }

    // Writer queue
    ctx->write_queue_capacity = pool_size * 2;
    ctx->write_queue = (FrameDataCollector**)calloc(ctx->write_queue_capacity, sizeof(FrameDataCollector*));
    if (!ctx->write_queue) {
        dc_destroy(ctx);
        return EB_ErrorInsufficientResources;
    }

    // Open HDF5 file
    ctx->hdf5_file = hdf5_writer_init(output_path, pic_width, pic_height,
                                       bit_depth, b64_total_count, sb_total_count);
    if (ctx->hdf5_file < 0) {
        SVT_ERROR("DC: Failed to create HDF5 file: %s\n", output_path);
        dc_destroy(ctx);
        return EB_ErrorBadParameter;
    }

    // Start writer thread
    ctx->writer_thread_exit = 0;
    ctx->writer_thread_handle = svt_create_thread(dc_writer_thread, ctx);
    if (!ctx->writer_thread_handle) {
        dc_destroy(ctx);
        return EB_ErrorInsufficientResources;
    }

    SVT_LOG("DC: Data collection initialized. Output: %s, Pool: %u, "
            "B64s: %u, SBs: %u, Raw: %s\n",
            output_path, pool_size, b64_total_count, sb_total_count,
            ctx->capture_raw_frames ? "ON" : "OFF");

    *ctx_out = ctx;
    return EB_ErrorNone;
}

void dc_destroy(DataCollectionContext* ctx) {
    if (!ctx)
        return;

    // Signal writer thread to exit
    if (ctx->writer_thread_handle) {
        ctx->writer_thread_exit = 1;
        if (ctx->write_queue_semaphore)
            svt_post_semaphore(ctx->write_queue_semaphore);
        svt_destroy_thread(ctx->writer_thread_handle);
        ctx->writer_thread_handle = NULL;
    }

    // Close HDF5
    if (ctx->hdf5_file >= 0) {
        hdf5_writer_close(ctx->hdf5_file);
        ctx->hdf5_file = -1;
    }

    SVT_LOG("DC: Data collection finished. Frames written: %llu, dropped: %llu\n",
            (unsigned long long)ctx->frames_written,
            (unsigned long long)ctx->frames_dropped);

    // Free collectors
    if (ctx->collectors) {
        for (uint32_t i = 0; i < ctx->pool_size; i++) {
            FrameDataCollector* fc = &ctx->collectors[i];
            if (fc->completion_mutex)
                svt_destroy_mutex(fc->completion_mutex);
            free(fc->me_data);
            free(fc->partition_data);
            free(fc->raw_luma_copy);
        }
        free(ctx->collectors);
    }

    free(ctx->write_queue);

    if (ctx->pool_mutex)
        svt_destroy_mutex(ctx->pool_mutex);
    if (ctx->write_queue_mutex)
        svt_destroy_mutex(ctx->write_queue_mutex);
    if (ctx->write_queue_semaphore)
        svt_destroy_semaphore(ctx->write_queue_semaphore);

    free(ctx);
}

// ---------- Hook API ----------

FrameDataCollector* dc_get_collector(DataCollectionContext* ctx, uint64_t picture_number) {
    if (!ctx || ctx->writer_thread_exit)
        return NULL;

    uint32_t idx = (uint32_t)(picture_number % ctx->pool_size);
    FrameDataCollector* fc = &ctx->collectors[idx];

    // Check if already assigned to this picture
    svt_block_on_mutex(ctx->pool_mutex);
    if (fc->in_use && fc->picture_number == picture_number) {
        svt_release_mutex(ctx->pool_mutex);
        return fc;
    }

    // If slot is occupied by another picture, drop
    if (fc->in_use) {
        svt_release_mutex(ctx->pool_mutex);
        return NULL;
    }

    // Initialize for this picture
    fc->in_use         = 1;
    fc->picture_number = picture_number;
    fc->me_complete    = 0;
    fc->enc_complete   = 0;
    fc->metadata_set   = 0;
    fc->ready_for_write = 0;
    fc->raw_luma_copy  = NULL;

    // Clear data arrays
    memset(fc->me_data, 0, ctx->b64_total_count * sizeof(DcMeData));
    memset(fc->partition_data, 0, ctx->sb_total_count * sizeof(DcPartitionData));

    svt_release_mutex(ctx->pool_mutex);
    return fc;
}

void dc_record_frame_metadata(FrameDataCollector* collector,
                              const DcFrameMetadata* metadata) {
    if (!collector)
        return;
    memcpy(&collector->metadata, metadata, sizeof(DcFrameMetadata));
    collector->metadata_set = 1;
}

void dc_record_me_sb(FrameDataCollector* collector,
                     uint32_t sb_index,
                     const DcMeData* me_data) {
    if (!collector || sb_index >= collector->me_sbs_total)
        return;
    // Lock-free: each sb_index is written by exactly one ME thread
    memcpy(&collector->me_data[sb_index], me_data, sizeof(DcMeData));
}

void dc_record_partition_sb(FrameDataCollector* collector,
                            uint32_t sb_index,
                            const DcPartitionData* partition_data) {
    if (!collector || sb_index >= collector->enc_sbs_total)
        return;
    // Lock-free: each sb_index is written by exactly one EncDec thread
    memcpy(&collector->partition_data[sb_index], partition_data, sizeof(DcPartitionData));
}

// Helper to enqueue completed collector to writer
static void dc_enqueue_for_write(DataCollectionContext* ctx, FrameDataCollector* collector) {
    svt_block_on_mutex(ctx->write_queue_mutex);
    uint32_t next_tail = (ctx->write_queue_tail + 1) % ctx->write_queue_capacity;
    if (next_tail == ctx->write_queue_head) {
        // Queue full, drop frame
        svt_release_mutex(ctx->write_queue_mutex);
        ctx->frames_dropped++;
        return;
    }
    ctx->write_queue[ctx->write_queue_tail] = collector;
    ctx->write_queue_tail = next_tail;
    svt_release_mutex(ctx->write_queue_mutex);
    svt_post_semaphore(ctx->write_queue_semaphore);
}

void dc_signal_me_complete(DataCollectionContext* ctx,
                           FrameDataCollector* collector) {
    if (!collector || !ctx)
        return;

    svt_block_on_mutex(collector->completion_mutex);
    collector->me_complete = 1;
    uint8_t ready = collector->me_complete && collector->enc_complete && collector->metadata_set;
    if (ready)
        collector->ready_for_write = 1;
    svt_release_mutex(collector->completion_mutex);

    if (ready)
        dc_enqueue_for_write(ctx, collector);
}

void dc_signal_enc_complete(DataCollectionContext* ctx,
                            FrameDataCollector* collector) {
    if (!collector || !ctx)
        return;

    svt_block_on_mutex(collector->completion_mutex);
    collector->enc_complete = 1;
    uint8_t ready = collector->me_complete && collector->enc_complete && collector->metadata_set;
    if (ready)
        collector->ready_for_write = 1;
    svt_release_mutex(collector->completion_mutex);

    if (ready)
        dc_enqueue_for_write(ctx, collector);
}

#endif // ENABLE_DATA_COLLECTION
