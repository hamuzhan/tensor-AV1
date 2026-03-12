/*
 * Helper to extract motion estimation data from MeContext
 * into the DcMeData struct for data collection.
 */

#ifndef SVT_ME_DATA_EXTRACT_H
#define SVT_ME_DATA_EXTRACT_H

#ifdef ENABLE_DATA_COLLECTION

#include "data_collector.h"

// Forward declare types from encoder
struct MeContext;
struct PictureParentControlSet;

#ifdef __cplusplus
extern "C" {
#endif

// Extract ME data from the MeContext after motion estimation is complete for a b64.
void dc_extract_me_data(const struct MeContext* me_ctx,
                        const struct PictureParentControlSet* pcs,
                        DcMeData* out,
                        uint16_t b64_origin_x,
                        uint16_t b64_origin_y);

#ifdef __cplusplus
}
#endif

#endif // ENABLE_DATA_COLLECTION
#endif // SVT_ME_DATA_EXTRACT_H
