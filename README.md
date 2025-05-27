# mzAPO 3d graphics development toolkit

![anim](misc/anim.gif)

## Building

### C code
```
clang demo.c
```

### Shaders
The desktop sokol-based backend uses a single tiny GPU shader to display
our CPU-rendered texture to the screen.

The shader is compiled using `sokol-shdc` from [sokol-tools](https://github.com/floooh/sokol-tools-bin).

Clone the repo and use the following command (with the correct `sokol-shdc` path depending on your platform):
```
sokol-tools-bin\bin\win32\sokol-shdc.exe --input shaders.glsl --output shaders.h --slang glsl430
```