#pragma once

#include <stdint.h>

// How many "real screen pixels" is one draw pixel.
// Higher the number, lower the resolution.
// Useful to improve performance, since pixel fill rate tends to be the bottleneck.
#ifndef MZR_PIXEL_SCALE
#define MZR_PIXEL_SCALE 1
#endif

#define MZR_PACKED_POS_MAX UINT16_MAX

#define MZR_BASE_RESOLUTION_X 480
#define MZR_BASE_RESOLUTION_Y 320
#define MZR_RESOLUTION_X (MZR_BASE_RESOLUTION_X / MZR_PIXEL_SCALE)
#define MZR_RESOLUTION_Y (MZR_BASE_RESOLUTION_Y / MZR_PIXEL_SCALE)

typedef struct mzr_mesh_ {
    int32_t     num_vertices;
    // Normalized integer positions mapped into the bounding box
    float*      vert_positions;
} mzr_mesh_t;

void mzr_initialize();
void mzr_shutdown();
void mzr_clear_depth(float far);

void mzr_draw_mesh(
    uint16_t*       framebuffer,
    mzr_mesh_t      mesh,
    const float     mvp_data[16], // column-major model-view-projection matrix
    float           color_r,
    float           color_g,
    float           color_b
);