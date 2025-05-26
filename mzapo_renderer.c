#include "mzapo_renderer.h"
#include "vmath.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define EDGE_STEP_X 2
#define EDGE_STEP_Y 2

#define MZR_DEPTH_TILE_X 8
#define MZR_DEPTH_TILE_Y 8
#define MZR_DEPTH_BUFFER_TILES_X (MZR_RESOLUTION_X / MZR_DEPTH_TILE_X)
#define MZR_DEPTH_BUFFER_TILES_Y (MZR_RESOLUTION_Y / MZR_DEPTH_TILE_Y)

typedef struct mzr_depth_tile_ {
    float data[MZR_DEPTH_TILE_X][MZR_DEPTH_TILE_Y];
} mzr_depth_tile_t;

typedef struct mzr_state_ {
    mzr_depth_tile_t* depth_buffer;
} mzr_state_t;

static mzr_state_t _mzr_state;

typedef struct edge_ {
    vec4i step_x;
    vec4i step_y;
} edge_t;

static inline vec4i edge_init(edge_t* edge, int2 v0, int2 v1, int2 origin) {
    // Edge setup
    int a = v0.y - v1.y;
    int b = v1.x - v0.x;
    int c = v0.x * v1.y - v0.y * v1.x;

    // Step deltas
    edge->step_x = vec4i_make_scalar(a * EDGE_STEP_X);
    edge->step_y = vec4i_make_scalar(b * EDGE_STEP_Y);

    // x/y values for initial pixel block
    vec4i x = vec4i_add(vec4i_make_scalar(origin.x), vec4i_make(0, 0, 1, 1));
    vec4i y = vec4i_add(vec4i_make_scalar(origin.y), vec4i_make(0, 1, 0, 1));

    // Edge function values at origin
    return vec4i_add(
        vec4i_add(
            vec4i_mul(vec4i_make_scalar(a), x),
            vec4i_mul(vec4i_make_scalar(b), y)),
        vec4i_make_scalar(c));
}


static inline float edge_func(vec4 a, vec4 b, vec4 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

void mzr_initialize() {
    _mzr_state.depth_buffer = calloc(sizeof(mzr_depth_tile_t), MZR_DEPTH_BUFFER_TILES_X * MZR_DEPTH_BUFFER_TILES_Y);
}

void mzr_shutdown() {

}

void mzr_clear_depth(float far) {
    for(int i = 0; i < MZR_DEPTH_BUFFER_TILES_X * MZR_DEPTH_BUFFER_TILES_Y; i++) {
        for(int x = 0; x < MZR_DEPTH_TILE_X; x++) {
            for (int y = 0; y < MZR_DEPTH_TILE_Y; y++) {
                _mzr_state.depth_buffer[i].data[x][y] = far;
            }
        }
    }
}

static inline float _read_depth_buffer(int x, int y) {
    int tile_x = x / MZR_DEPTH_TILE_X;
    int tile_y = y / MZR_DEPTH_TILE_Y;
    return _mzr_state.depth_buffer[tile_x + tile_y * MZR_DEPTH_BUFFER_TILES_X]
        .data[x % MZR_DEPTH_TILE_X][y % MZR_DEPTH_TILE_Y];
}


static inline uint16_t _vec4_pack_rgb565_simple(float r, float g, float b) {
    uint16_t color = 0;
    color |= ((uint16_t)(r)) << 11;
    color |= ((uint16_t)(g)) << 5;
    color |= ((uint16_t)(b)) << 0;
    return color;
}


void mzr_draw_mesh(
    uint16_t* const framebuffer,
    const mzr_mesh_t mesh,
    const mzr_texture_t texture,
    const float mvp_data[16],
    const float  color_r,
    const float  color_g,
    const float  color_b
) {
    vec4 bounds_min = vec4_make(
        mesh.bounds_min_x,
        mesh.bounds_min_y,
        mesh.bounds_min_z,
        0.0f);
    vec4 bounds_scale = vec4_make(
        (mesh.bounds_max_x - mesh.bounds_min_x) / (float)MZR_PACKED_POS_MAX,
        (mesh.bounds_max_y - mesh.bounds_min_y) / (float)MZR_PACKED_POS_MAX,
        (mesh.bounds_max_z - mesh.bounds_min_z) / (float)MZR_PACKED_POS_MAX,
        1.0f);

    mat4 mvp = {0};
    memcpy(mvp.data, mvp_data, sizeof(mat4));

    int tri_num = mesh.num_indices / 3;
    printf("Drawing mesh with %i triangles\n", tri_num);
    for(int tri_index = 0; tri_index < tri_num; tri_index++) {
        uint32_t index0 = mesh.indices[tri_index * 3 + 0];
        uint32_t index1 = mesh.indices[tri_index * 3 + 1];
        uint32_t index2 = mesh.indices[tri_index * 3 + 2];

        printf("Triangle %i: %i/%i/%i\n", tri_index, index0, index1, index2);

        vec4 v0 = vec4_make(
            mesh.vert_positions[index0 * 3 + 0],
            mesh.vert_positions[index0 * 3 + 1],
            mesh.vert_positions[index0 * 3 + 2],
            1.0);
        vec4 v1 = vec4_make(
            mesh.vert_positions[index1 * 3 + 0],
            mesh.vert_positions[index1 * 3 + 1],
            mesh.vert_positions[index1 * 3 + 2],
            1.0);
        vec4 v2 = vec4_make(
            mesh.vert_positions[index2 * 3 + 0],
            mesh.vert_positions[index2 * 3 + 1],
            mesh.vert_positions[index2 * 3 + 2],
            1.0);

        v0 = vec4_add(bounds_min, vec4_mul(v0, bounds_scale));
        v1 = vec4_add(bounds_min, vec4_mul(v1, bounds_scale));
        v2 = vec4_add(bounds_min, vec4_mul(v2, bounds_scale));

        float2 uv0 = float2_make(
            (float)mesh.vert_uvs[index0 * 2 + 0] / (float)MZR_PACKED_UV_MAX,
            (float)mesh.vert_uvs[index0 * 2 + 1] / (float)MZR_PACKED_UV_MAX);
        float2 uv1 = float2_make(
            (float)mesh.vert_uvs[index1 * 2 + 0] / (float)MZR_PACKED_UV_MAX,
            (float)mesh.vert_uvs[index1 * 2 + 1] / (float)MZR_PACKED_UV_MAX);
        float2 uv2 = float2_make(
            (float)mesh.vert_uvs[index2 * 2 + 0] / (float)MZR_PACKED_UV_MAX,
            (float)mesh.vert_uvs[index2 * 2 + 1] / (float)MZR_PACKED_UV_MAX);

        vec4 normal = vec3_normalize(vec3_cross(
            vec4_sub(v1, v0),
            vec4_sub(v2, v0)));

        v0 = mat4_mul_vec4(v0, mvp);
        v1 = mat4_mul_vec4(v1, mvp);
        v2 = mat4_mul_vec4(v2, mvp);

        v0 = vec4_scale(v0, 1.0 / v0.w);
        v1 = vec4_scale(v1, 1.0 / v1.w);
        v2 = vec4_scale(v2, 1.0 / v2.w);

        float area = edge_func(v0, v1, v2);
        // Backface culling
        if (area < 0.0) {
            continue;
        }

        vec4 inv_area = vec4_make_scalar(1.0 / area);

        // float linear_z = near * far / (far - v0.z * (far - near));

        // TODO: move into mvp?
        v0.x = (v0.x * 0.5 + 0.5) * MZR_RESOLUTION_X;
        v0.y = (v0.y * -0.5 + 0.5) * MZR_RESOLUTION_Y;
        v1.x = (v1.x * 0.5 + 0.5) * MZR_RESOLUTION_X;
        v1.y = (v1.y * -0.5 + 0.5) * MZR_RESOLUTION_Y;
        v2.x = (v2.x * 0.5 + 0.5) * MZR_RESOLUTION_X;
        v2.y = (v2.y * -0.5 + 0.5) * MZR_RESOLUTION_Y;

        int min_x = int_min(v0.x, int_min(v1.x, v2.x));
        int max_x = int_max(v0.x, int_max(v1.x, v2.x));
        int min_y = int_min(v0.y, int_min(v1.y, v2.y));
        int max_y = int_max(v0.y, int_max(v1.y, v2.y));

        min_x = int_max(0, min_x);
        min_y = int_max(0, min_y);
        max_x = int_min(max_x, MZR_RESOLUTION_X - 1);
        max_y = int_min(max_y, MZR_RESOLUTION_Y - 1);

        edge_t e12;
        edge_t e20;
        edge_t e01;

        // Make sure quads are aligned to multiples of 2
        min_x = 2 * (min_x >> 1);
        min_y = 2 * (min_y >> 1);
        max_x = 2 * ((max_x + 1) >> 1);
        max_y = 2 * ((max_y + 1) >> 1);

        vec4i w0_row = edge_init(&e12, int2_make(v1.x, v1.y), int2_make(v2.x, v2.y), int2_make(min_x, min_y));
        vec4i w1_row = edge_init(&e20, int2_make(v2.x, v2.y), int2_make(v0.x, v0.y), int2_make(min_x, min_y));
        vec4i w2_row = edge_init(&e01, int2_make(v0.x, v0.y), int2_make(v1.x, v1.y), int2_make(min_x, min_y));

        for(int y = min_y; y <= max_y; y += EDGE_STEP_Y) {
            vec4i w0 = w0_row;
            vec4i w1 = w1_row;
            vec4i w2 = w2_row;

            for(int x = min_x; x <= max_x; x += EDGE_STEP_X,
                    w0 = vec4i_add(w0, e12.step_x),
                    w1 = vec4i_add(w1, e20.step_x),
                    w2 = vec4i_add(w2, e01.step_x)) {
                vec4i mask = vec4i_or(w0, vec4i_or(w1, w2));

                // Test sign bit
                mask = vec4i_and(vec4i_make_scalar(0x80000000), mask);
                mask = vec4i_equal(mask, vec4i_make_scalar(0));

                // Early out if the quad if empty
                // (if any w0/1/2 >= 0)
                if(vec4i_test_all_zeros(mask)) {
                    continue;
                }

                vec4 w0a = vec4_mul(vec4i_to_vec4(w0), inv_area);
                vec4 w1a = vec4_mul(vec4i_to_vec4(w1), inv_area);
                vec4 w2a = vec4_mul(vec4i_to_vec4(w2), inv_area);
                vec4 one_over_z =
                    vec4_add(
                        vec4_mul(w0a, vec4_make_scalar(1.0 / v0.z)),
                        vec4_add(
                            vec4_mul(w1a, vec4_make_scalar(1.0 / v1.z)),
                            vec4_mul(w2a, vec4_make_scalar(1.0 / v2.z))));
                vec4 bary_scale = vec4_div(vec4_make_scalar(1.0f), one_over_z);
                w0a = vec4_mul(w0a, bary_scale);
                w1a = vec4_mul(w1a, bary_scale);
                w2a = vec4_mul(w2a, bary_scale);

                // Interpolate the depth
                const vec4 depth =
                    vec4_add(
                        vec4_mul(w0a, vec4_make_scalar(v0.z)),
                        vec4_add(
                            vec4_mul(w1a, vec4_make_scalar(v1.z)),
                            vec4_mul(w2a, vec4_make_scalar(v2.z))));

                int tile_x = x / MZR_DEPTH_TILE_X;
                int tile_y = y / MZR_DEPTH_TILE_Y;
                int tile_sub_x = x % MZR_DEPTH_TILE_X;
                int tile_sub_y = y % MZR_DEPTH_TILE_Y;
                mzr_depth_tile_t* tile = &_mzr_state.depth_buffer[tile_x + tile_y * MZR_DEPTH_BUFFER_TILES_X];

                const vec4 min_depth = vec4_min(
                    depth,
                    vec4_make(
                        tile->data[tile_sub_x + 0][tile_sub_y + 0],
                        tile->data[tile_sub_x + 0][tile_sub_y + 1],
                        tile->data[tile_sub_x + 1][tile_sub_y + 0],
                        tile->data[tile_sub_x + 1][tile_sub_y + 1]));

                mask = vec4i_and(mask, vec4_equal(depth, min_depth));

                // Early out
                if(vec4i_test_all_zeros(mask)) {
                    continue;
                }

                if (mask.x != 0) tile->data[tile_sub_x + 0][tile_sub_y + 0] = min_depth.x;
                if (mask.y != 0) tile->data[tile_sub_x + 0][tile_sub_y + 1] = min_depth.y;
                if (mask.z != 0) tile->data[tile_sub_x + 1][tile_sub_y + 0] = min_depth.z;
                if (mask.w != 0) tile->data[tile_sub_x + 1][tile_sub_y + 1] = min_depth.w;

                vec4 uv_x =
                    vec4_add(
                        vec4_mul(w0a, vec4_make_scalar(uv0.x)),
                        vec4_add(
                            vec4_mul(w1a, vec4_make_scalar(uv1.x)),
                            vec4_mul(w2a, vec4_make_scalar(uv2.x))));
                vec4 uv_y =
                    vec4_add(
                        vec4_mul(w0a, vec4_make_scalar(uv0.y)),
                        vec4_add(
                            vec4_mul(w1a, vec4_make_scalar(uv1.y)),
                            vec4_mul(w2a, vec4_make_scalar(uv2.y))));


                vec4i texcoord_x = vec4i_max(
                    vec4i_make_scalar(0),
                    vec4i_min(
                        vec4i_make_scalar(texture.size_x - 1),
                        vec4_to_vec4i(vec4_mul(uv_x, vec4_make_scalar(texture.size_x)))));
                vec4i texcoord_y = vec4i_max(
                    vec4i_make_scalar(0),
                    vec4i_min(
                        vec4i_make_scalar(texture.size_y - 1),
                        vec4_to_vec4i(vec4_mul(uv_y, vec4_make_scalar(texture.size_y)))));

                uint16_t tex_colors[4] = {
                    texture.pixels[texcoord_x.x + texcoord_y.x * texture.size_x],
                    texture.pixels[texcoord_x.y + texcoord_y.y * texture.size_x],
                    texture.pixels[texcoord_x.z + texcoord_y.z * texture.size_x],
                    texture.pixels[texcoord_x.w + texcoord_y.w * texture.size_x],
                };

                // vec4 col_r = uv_x;
                // vec4 col_g = uv_y;
                // vec4 col_b = vec4_make_scalar(0.0); // color_b); //0.5);

                // // float diffuse = 0.5 + 0.5f * float_max(0.0, vec3_dot(normal, vec4_make(0.3, 0.8, 0.5, 0.0)));
                // // col_r = vec4_mul(col_r, vec4_make_scalar(diffuse));
                // // col_g = vec4_mul(col_g, vec4_make_scalar(diffuse));
                // // col_b = vec4_mul(col_b, vec4_make_scalar(diffuse));

                // // col_b = vec4_make_scalar(
                // //     0.1f * (1.0f - float_max(0.0, normal.y) * float_max(0.0, normal.y))
                // // );

                // // // Pack colors to RGB565
                // col_r = vec4_scale(col_r, 31.0);
                // col_g = vec4_scale(col_g, 63.0);
                // col_b = vec4_scale(col_b, 31.0);

                // uint16_t packed_col0 = _vec4_pack_rgb565_simple(col_r.x, col_g.x, col_b.x);
                // uint16_t packed_col1 = _vec4_pack_rgb565_simple(col_r.y, col_g.y, col_b.y);
                // uint16_t packed_col2 = _vec4_pack_rgb565_simple(col_r.z, col_g.z, col_b.z);
                // uint16_t packed_col3 = _vec4_pack_rgb565_simple(col_r.w, col_g.w, col_b.w);

                // TODO: blend write to quad-tiled framebuffer
                if (mask.x != 0) framebuffer[(x + 0) + (y + 0) * MZR_RESOLUTION_X] = tex_colors[0];
                if (mask.y != 0) framebuffer[(x + 0) + (y + 1) * MZR_RESOLUTION_X] = tex_colors[1];
                if (mask.z != 0) framebuffer[(x + 1) + (y + 0) * MZR_RESOLUTION_X] = tex_colors[2];
                if (mask.w != 0) framebuffer[(x + 1) + (y + 1) * MZR_RESOLUTION_X] = tex_colors[3];
            }

            // Step down
            w0_row = vec4i_add(w0_row, e12.step_y);
            w1_row = vec4i_add(w1_row, e20.step_y);
            w2_row = vec4i_add(w2_row, e01.step_y);
        }
    }
}

