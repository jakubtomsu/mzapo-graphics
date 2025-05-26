#pragma once

#include <stdint.h>

// How many "real screen pixels" is one draw pixel.
// Higher the number, lower the resolution.
// Useful to improve performance, since pixel fill rate tends to be the bottleneck.
#ifndef MZR_PIXEL_SCALE
#define MZR_PIXEL_SCALE 1
#endif

#define MZR_PACKED_POS_MAX UINT16_MAX
#define MZR_PACKED_UV_MAX UINT8_MAX

#define MZR_BASE_RESOLUTION_X 480
#define MZR_BASE_RESOLUTION_Y 320
#define MZR_RESOLUTION_X (MZR_BASE_RESOLUTION_X / MZR_PIXEL_SCALE)
#define MZR_RESOLUTION_Y (MZR_BASE_RESOLUTION_Y / MZR_PIXEL_SCALE)

typedef struct mzr_mesh_t {
    int32_t     num_indices;
    int32_t     num_vertices;
    float       bounds_min_x;
    float       bounds_min_y;
    float       bounds_min_z;
    float       bounds_max_x;
    float       bounds_max_y;
    float       bounds_max_z;

    uint32_t*   indices;
    // Normalized integer positions mapped into the bounding box
    uint16_t*   vert_positions;
    uint8_t*    vert_uvs;
} mzr_mesh_t;

typedef struct mzr_texture_t {
    int32_t     size_x;
    int32_t     size_y;
    uint16_t*   pixels;
} mzr_texture_t;

void mzr_initialize();
void mzr_shutdown();
void mzr_clear_depth(float far);

void mzr_draw_mesh(
    uint16_t*       framebuffer,
    mzr_mesh_t      mesh,
    mzr_texture_t   texture,
    const float     mvp_data[16], // column-major model-view-projection matrix
    float           color_r,
    float           color_g,
    float           color_b
);