/*
 * Extracts motion estimation data from MeContext into DcMeData
 * for the data collection pipeline.
 */

#ifdef ENABLE_DATA_COLLECTION

#include "me_data_extract.h"
#include "me_context.h"
#include "pcs.h"

#include <string.h>

void dc_extract_me_data(const MeContext* me_ctx,
                        const PictureParentControlSet* pcs,
                        DcMeData* out,
                        uint16_t b64_origin_x,
                        uint16_t b64_origin_y) {
    if (!me_ctx || !out)
        return;

    memset(out, 0, sizeof(DcMeData));

    out->sb_origin_x = b64_origin_x;
    out->sb_origin_y = b64_origin_y;
    out->sb_width    = (uint16_t)me_ctx->b64_width;
    out->sb_height   = (uint16_t)me_ctx->b64_height;
    out->valid       = 1;

    // Number of active references per list
    out->num_refs[0] = me_ctx->num_of_ref_pic_to_search[0];
    out->num_refs[1] = me_ctx->num_of_ref_pic_to_search[1];

    // Copy SADs and MVs for all square PUs (85 entries)
    for (int list = 0; list < DC_MAX_REF_LISTS; list++) {
        uint8_t num_refs = out->num_refs[list];
        if (num_refs > DC_MAX_REFS_PER_LIST)
            num_refs = DC_MAX_REFS_PER_LIST;
        for (int ref = 0; ref < num_refs; ref++) {
            for (int pu = 0; pu < DC_SQUARE_PU_COUNT; pu++) {
                out->best_sad[list][ref][pu] = me_ctx->p_sb_best_sad[list][ref][pu];
                // ME stores MVs as packed uint32_t: lower 16 = x, upper 16 = y
                uint32_t packed_mv = me_ctx->p_sb_best_mv[list][ref][pu];
                out->best_mv[list][ref][pu].x = (int16_t)(packed_mv & 0xFFFF);
                out->best_mv[list][ref][pu].y = (int16_t)(packed_mv >> 16);
            }
        }
    }
}

#endif // ENABLE_DATA_COLLECTION
