// clang demo.c -std=c99 -lGL -lX11 -lXrandr -lXi -lXcursor -pthread -lm -ldl -o demo.out


#include <stdint.h>
#include <stdio.h>

#define SOKOL_IMPL
#ifndef _SAPP_WIN32
#define SOKOL_GLCORE
#endif
#define SOKOL_NO_ENTRY

#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"
#include "shaders.h"
#include "mzapo_renderer.h"
#include "vmath.h"

#define WINDOW_SCALE 2
#define ARRAY_LEN(arr) (sizeof(arr) / sizeof(arr[0]))

typedef struct {
  sg_pass_action pass_action;
  sg_pipeline pip;
  sg_bindings bind;
  sg_image img;
  sg_sampler smp;
  int frame_index;
} program_state_t;

static program_state_t g_state;

void _app_init(void) {
  sg_setup(&(sg_desc){
      .environment = sglue_environment(),
      .logger.func = slog_func,
  });

  srand(2938492834);

  // a vertex buffer
  float vertices[] = {
      -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
  };
  g_state.bind.vertex_buffers[0] = sg_make_buffer(
      &(sg_buffer_desc){.data = SG_RANGE(vertices), .label = "quad-vertices"});

  // an index buffer with 2 triangles
  uint16_t indices[] = {0, 1, 2, 0, 2, 3};
  g_state.bind.index_buffer =
      sg_make_buffer(&(sg_buffer_desc){.type = SG_BUFFERTYPE_INDEXBUFFER,
                                       .data = SG_RANGE(indices),
                                       .label = "quad-indices"});

    g_state.img = sg_make_image(&(sg_image_desc){
        .label = "framebuffer-image",
        .type = SG_IMAGETYPE_2D,
        .usage = SG_USAGE_STREAM,
        .pixel_format = SG_PIXELFORMAT_RGBA8,
        .width = MZR_RESOLUTION_X,
        .height = MZR_RESOLUTION_Y,
        .sample_count = 1,
        .num_mipmaps = 1,
        .num_slices = 1,
    });

    g_state.smp = sg_make_sampler(&(sg_sampler_desc){
        .label = "framebuffer-sampler",
        .min_filter = SG_FILTER_NEAREST,
        .mag_filter = SG_FILTER_NEAREST,
        .mipmap_filter = SG_FILTER_NEAREST,
        .wrap_u = SG_WRAP_REPEAT,
        .wrap_v = SG_WRAP_REPEAT,
        .wrap_w = SG_WRAP_REPEAT,
    });

    g_state.bind.images[IMG_tex] = g_state.img;
    g_state.bind.samplers[SMP_smp] = g_state.smp;

  sg_shader shd = sg_make_shader(quad_shader_desc(sg_query_backend()));

  g_state.pip = sg_make_pipeline(&(sg_pipeline_desc){
      .shader = shd,
      .index_type = SG_INDEXTYPE_UINT16,
      .layout = {.attrs =
                     {
                         [ATTR_quad_position].format = SG_VERTEXFORMAT_FLOAT2,
                     }},
      .label = "quad-pipeline"});

  // default pass action, no need to clear
  g_state.pass_action =
      (sg_pass_action){.colors[0] = {.load_action = SG_LOADACTION_DONTCARE}};

    mzr_initialize();
}

void _app_cleanup(void) {
    mzr_shutdown();
    sg_shutdown();
}

void _app_event(const sapp_event *event) {}


// The emulation layer has to convert to a different format before uploading to the GPU.
uint32_t g_framebuf_rgba8[MZR_RESOLUTION_X * MZR_RESOLUTION_Y];
uint16_t g_framebuf_rgb565[MZR_RESOLUTION_X * MZR_RESOLUTION_Y];

uint32_t cube_indices[] = {
    0, 1, 2,
    1, 3, 2,
};

uint16_t cube_vert_positions[] = {
    0, 0, 0,
    MZR_PACKED_POS_MAX, 0, 0,
    0, 0, MZR_PACKED_POS_MAX,
    MZR_PACKED_POS_MAX, 0, MZR_PACKED_POS_MAX,
};

uint8_t cube_vert_uvs[] = {
    0, 0,
    MZR_PACKED_UV_MAX, 0,
    0, MZR_PACKED_UV_MAX,
    MZR_PACKED_UV_MAX, MZR_PACKED_UV_MAX,
};

uint16_t tex_pixels[] = {
    0x00ff, 0x00ff, 0x00ff, 0x00ff,
    0x00ff, 0x0000, 0xffff, 0x00ff,
    0x00ff, 0xffff, 0x0000, 0x00ff,
    0x00ff, 0x00ff, 0x00ff, 0x00ff,
};

mat4 camera_calc_view_projection(
    float camera_dist,
    float camera_yaw,
    float camera_pitch
) {
    float fov = 70.0f;
    float near = 0.01f;
    float far = 100.0f;
    mat4 proj = mat4_perspective(fov, (float)MZR_RESOLUTION_X / (float)MZR_RESOLUTION_Y, near, far);

    vec4 camera_pos = vec4_make(0, 0, camera_dist, 0);
    mat4 view = mat4_mul(
        mat4_translate(camera_pos),
        mat4_mul(
            mat4_rotate(vec4_make(1, 0, 0, 0), camera_pitch),
            mat4_rotate(vec4_make(0, 1, 0, 0), camera_yaw)));

    mat4 mvp = mat4_mul(proj, view);

    return mvp;
}

void _app_frame(void) {
    // g_framebuf_rgb565[(g_state.frame_index) % (MZR_RESOLUTION_X * MZR_RESOLUTION_Y)] = vec4_pack_rgb565_simple(1.0, 1.0, 0.0);
    for (int i = 0; i < MZR_RESOLUTION_X * MZR_RESOLUTION_Y; i++) {
        g_framebuf_rgb565[i] = 0;
    }

    mzr_clear_depth(1e20f);

    mzr_mesh_t mesh = {0};
    mesh.bounds_min_x = -1.0f;
    mesh.bounds_min_y = -1.0f;
    mesh.bounds_min_z = -1.0f;
    mesh.bounds_max_x =  1.0f;
    mesh.bounds_max_y =  1.0f;
    mesh.bounds_max_z =  1.0f;
    mesh.num_indices = ARRAY_LEN(cube_indices);
    mesh.indices = cube_indices;
    mesh.num_vertices = ARRAY_LEN(cube_vert_positions) / 3;
    mesh.vert_positions = cube_vert_positions;
    mesh.vert_uvs = cube_vert_uvs;

    float camera_yaw = (float)g_state.frame_index * 0.01;
    float camera_dist = 4.0f + sinf((float)g_state.frame_index * 0.005);
    mat4 mvp = {0};
    mvp = camera_calc_view_projection(camera_dist, camera_yaw, 0.5);

    mzr_texture_t tex = {0};
    tex.size_x = 4;
    tex.size_y = 4;
    tex.pixels = tex_pixels;

    mzr_draw_mesh(g_framebuf_rgb565, mesh, tex, mvp.data, 1.0f, 0.0f, 1.0f);

    for (int i = 0; i < MZR_RESOLUTION_X * MZR_RESOLUTION_Y; i++) {
        uint16_t src = g_framebuf_rgb565[i];

        uint32_t dst = 0;
        dst |= (((uint32_t)(src >> 11) % 32) * (256 / 32)) <<  0;
        dst |= (((uint32_t)(src >>  5) % 64) * (256 / 64)) <<  8;
        dst |= (((uint32_t)(src >>  0) % 32) * (256 / 32)) << 16;

        g_framebuf_rgba8[i] = dst;
    }

    sg_image_data img_data = {0};
    img_data.subimage[0][0] = (sg_range){.ptr = &g_framebuf_rgba8[0], .size = sizeof(uint32_t) * MZR_RESOLUTION_X * MZR_RESOLUTION_Y};
    sg_update_image(g_state.img, &img_data);

    sg_begin_pass(&(sg_pass){
        .action = g_state.pass_action,
        .swapchain = sglue_swapchain()});
    sg_apply_pipeline(g_state.pip);
    sg_apply_bindings(&g_state.bind);
    sg_draw(0, 6, 1);
    sg_end_pass();
    sg_commit();

    g_state.frame_index += 1;
}

int main(int argc, char *argv[]) {
(void)argc;
  (void)argv;
  sapp_run(&(sapp_desc){
      .init_cb = _app_init,
      .frame_cb = _app_frame,
      .cleanup_cb = _app_cleanup,
      .event_cb = _app_event,
      .width  = MZR_BASE_RESOLUTION_X * WINDOW_SCALE,
      .height = MZR_BASE_RESOLUTION_Y * WINDOW_SCALE,
      .window_title = "mzAPO Emulation Layer",
      .icon.sokol_default = true,
      .logger.func = slog_func,
  });
}

#include "mzapo_renderer.c"