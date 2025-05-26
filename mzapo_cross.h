// A lightweight layer which abstracts over the most common features of mzAPO hardware,
// and implements them using Sokol libs for cross-platform testing support.
// The goal is to make testing and development easier.
// Supported platform include mainly Linux and Windows, but WASM should also work.
//
// Rendering is done using a framebuffer.
//
// Input is emulated with keyboard input.
//
// Audio and LED output is not supported.
#pragma once

#include <stdint.h>

#ifndef MZC_SOKOL
#define MZC_SOKOL
#endif

typedef void (*mzc_frame_callback_t)(mzc_frame_t frame);

typedef struct mzc_frame_t {
    uint32_t    frame_index;
    uint16_t*   framebuffer;
    uint32_t    knobs;
} mzc_frame_t;

void mzc_initialize(int pixel_scale);
void mzc_shutdown();

void mzc_run(mzc_frame_callback_t frame_cb);