#pragma once

// Vector Math
//
// Resources used when writing this code:
// - https://github.com/HandmadeMath/HandmadeMath/blob/master/HandmadeMath.h
// - https://github.com/simd-everywhere/simde
// - https://developer.arm.com/architectures/instruction-sets/intrinsics

// TODO: vectorize common ops!

#define PI 3.14159265358979323846264338327950288

#include <stdint.h>

#ifdef __ARM_NEON
#define VMATH_USE_NEON
#else
// no NEON!
#endif

#ifdef VMATH_USE_NEON
#include <arm_neon.h>
#endif

typedef union float2 {
    float elements[2];

    struct {
        float x;
        float y;
    };
} float2;

typedef union float3 {
    float elements[3];

    struct {
        float x;
        float y;
        float z;
    };
} float3;

typedef union int2 {
    int elements[2];

    struct {
        int x;
        int y;
    };
} int2;

typedef union vec4 {
    float elements[4];

    struct {
        float x;
        float y;
        float z;
        float w;
    };
#ifdef VMATH_USE_NEON
    float32x4_t neon_f32x4;
#endif
} vec4;

typedef union vec4i {
    int elements[4];

    struct {
        int x;
        int y;
        int z;
        int w;
    };
#ifdef VMATH_USE_NEON
    int32x4_t neon_s32x4;
    uint32x4_t neon_u32x4;
    int64x2_t neon_s64x2;
#endif
} vec4i;

typedef union mat4 {
    float data[16]; // flat elements
    float elements[4][4];
    vec4 columns[4];
} mat4;

static inline float2 float2_make(float x, float y) {
    float2 result = {0};
    result.x = x;
    result.y = y;
    return result;
}

static inline int2 int2_make(int x, int y) {
    int2 result = {0};
    result.x = x;
    result.y = y;
    return result;
}

static inline vec4 vec4_make(float x, float y, float z, float w) {
    vec4 result = {0};
    result.x = x;
    result.y = y;
    result.z = z;
    result.w = w;
    return result;
}

static inline vec4 vec4_make_scalar(float v) {
    vec4 result = {0};
    result.x = v;
    result.y = v;
    result.z = v;
    result.w = v;
    return result;
}


static inline vec4i vec4i_make(int x, int y, int z, int w) {
    vec4i result = {0};
    result.x = x;
    result.y = y;
    result.z = z;
    result.w = w;
    return result;
}

static inline vec4i vec4i_make_scalar(int v) {
    vec4i result = {0};
    result.x = v;
    result.y = v;
    result.z = v;
    result.w = v;
    return result;
}

static inline float vec3_dot(vec4 a, vec4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline vec4i vec4i_and(vec4i a, vec4i b) {
    vec4i result = {0};
#ifdef VMATH_USE_NEON
    result.neon_s32x4 = vandq_s32(a.neon_s32x4, b.neon_s32x4);
#else
    result.x = a.x & b.x;
    result.y = a.y & b.y;
    result.z = a.z & b.z;
    result.w = a.w & b.w;
#endif
    return result;
}

static inline void vec4i_print(const char* name, vec4i a) {
    printf("%s:{%x, %x, %x, %x}\n", name, a.x, a.y, a.z, a.w);
}

static inline vec4i vec4i_not(vec4i a) {
    vec4i result = {0};
#ifdef VMATH_USE_NEON
    result.neon_s32x4 = vmvnq_s32(a.neon_s32x4);
#else
    result.x = ~a.x;
    result.y = ~a.y;
    result.z = ~a.z;
    result.w = ~a.w;
#endif
    return result;
}

static inline vec4i vec4i_or(vec4i a, vec4i b) {
    vec4i result = {0};
#ifdef VMATH_USE_NEON
    result.neon_s32x4 = vorrq_s32(a.neon_s32x4, b.neon_s32x4);
#else
    result.x = a.x | b.x;
    result.y = a.y | b.y;
    result.z = a.z | b.z;
    result.w = a.w | b.w;
#endif
    return result;
}

static inline vec4i vec4i_add(vec4i a, vec4i b) {
    vec4i result = {0};
#ifdef VMATH_USE_NEON
    result.neon_s32x4 = vaddq_s32(a.neon_s32x4, b.neon_s32x4);
#else
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    result.w = a.w + b.w;
#endif
    return result;
}

static inline vec4i vec4i_mul(vec4i a, vec4i b) {
    vec4i result = {0};
#ifdef VMATH_USE_NEON
    result.neon_s32x4 = vmulq_s32(a.neon_s32x4, b.neon_s32x4);
#else
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    result.z = a.z * b.z;
    result.w = a.w * b.w;
#endif
    return result;
}

static inline vec4i vec4i_equal(vec4i a, vec4i b) {
    vec4i result = {0};
#ifdef VMATH_USE_NEON
    result.neon_u32x4 = vceqq_s32(a.neon_s32x4, b.neon_s32x4);
#else
    result.x = a.x == b.x ? 0xFFFFFFFF : 0;
    result.y = a.y == b.y ? 0xFFFFFFFF : 0;
    result.z = a.z == b.z ? 0xFFFFFFFF : 0;
    result.w = a.w == b.w ? 0xFFFFFFFF : 0;
#endif
    return result;
}


static inline vec4i vec4i_min(vec4i a, vec4i b) {
    vec4i result = {0};
#ifdef VMATH_USE_NEON
    result.neon_f32x4 = vminq_s32(a.neon_f32x4, b.neon_f32x4);
#else
    result.x = a.x < b.x ? a.x : b.x;
    result.y = a.y < b.y ? a.y : b.y;
    result.z = a.z < b.z ? a.z : b.z;
    result.w = a.w < b.w ? a.w : b.w;
#endif
    return result;
}

static inline vec4i vec4i_max(vec4i a, vec4i b) {
    vec4i result = {0};
#ifdef VMATH_USE_NEON
    result.neon_f32x4 = vmaxq_s32(a.neon_f32x4, b.neon_f32x4);
#else
    result.x = a.x > b.x ? a.x : b.x;
    result.y = a.y > b.y ? a.y : b.y;
    result.z = a.z > b.z ? a.z : b.z;
    result.w = a.w > b.w ? a.w : b.w;
#endif
    return result;
}


static inline int vec4i_test_all_zeros(vec4i a) {
// #ifdef VMATH_USE_NEON
//     return !(vgetq_lane_s64(a.neon_s64x2, 0) | vgetq_lane_s64(a.neon_s64x2, 1));
// #else
    return a.x == 0 && a.y == 0 && a.z == 0 && a.w == 0;
// #endif
}

// static inline vec4i vec4i_test(vec4i a, vec4i b) {
//     vec4i result = {0};
// #ifdef VMATH_USE_NEON
//     // int64x2_t s = vbicq_s64(b.neon_f32x4, a.neon_f32x4);
//     // result.neon_s32x4 = !(vgetq_lane_s64(s, 0) | vgetq_lane_s64(s, 1));
// #else
// #endif
//     return result;
// }


static inline float vec3_length(vec4 a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

static inline vec4 vec4_min(vec4 a, vec4 b) {
    vec4 result = {0};
#ifdef VMATH_USE_NEON
    result.neon_f32x4 = vminq_f32(a.neon_f32x4, b.neon_f32x4);
#else
    result.x = a.x < b.x ? a.x : b.x;
    result.y = a.y < b.y ? a.y : b.y;
    result.z = a.z < b.z ? a.z : b.z;
    result.w = a.w < b.w ? a.w : b.w;
#endif
    return result;
}

static inline vec4 vec4_max(vec4 a, vec4 b) {
    vec4 result = {0};
#ifdef VMATH_USE_NEON
    result.neon_f32x4 = vmaxq_f32(a.neon_f32x4, b.neon_f32x4);
#else
    result.x = a.x > b.x ? a.x : b.x;
    result.y = a.y > b.y ? a.y : b.y;
    result.z = a.z > b.z ? a.z : b.z;
    result.w = a.w > b.w ? a.w : b.w;
#endif
    return result;
}

static inline vec4i vec4_to_vec4i(vec4 a) {
    vec4i result = {0};
    result.x = (int)a.x;
    result.y = (int)a.y;
    result.z = (int)a.z;
    result.w = (int)a.w;
    return result;
}

static inline vec4 vec4i_to_vec4(vec4i a) {
    vec4 result = {0};
    result.x = (float)a.x;
    result.y = (float)a.y;
    result.z = (float)a.z;
    result.w = (float)a.w;
    return result;
}

static inline vec4 vec4_scale(vec4 a, float b) {
    vec4 result = {0};
    result.x = a.x * b;
    result.y = a.y * b;
    result.z = a.z * b;
    result.w = a.w * b;
    return result;
}

static inline float float_min(float a, float b) {
    return a < b ? a : b;
}

static inline float float_max(float a, float b) {
    return a > b ? a : b;
}

static inline float float_clamp(float a, float min, float max) {
    return float_max(min, float_min(max, a));
}

static inline int int_min(int a, int b) {
    return a < b ? a : b;
}

static inline int int_max(int a, int b) {
    return a > b ? a : b;
}

static inline vec4 vec3_normalize(vec4 a) {
    return vec4_scale(a, 1.0 / vec3_length(a));
}

static inline vec4 vec3_cross(vec4 a, vec4 b) {
    vec4 result = {0};
    result.x = (a.y * b.z) - (a.z * b.y);
    result.y = (a.z * b.x) - (a.x * b.z);
    result.z = (a.x * b.y) - (a.y * b.x);
    return result;
}

static inline vec4 vec4_add(vec4 a, vec4 b) {
    vec4 result = {0};
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    result.w = a.w + b.w;
    return result;
}

static inline vec4 vec4_sub(vec4 a, vec4 b) {
    vec4 result = {0};
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    result.w = a.w - b.w;
    return result;
}

static inline vec4 vec4_mul(vec4 a, vec4 b) {
    vec4 result = {0};
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    result.z = a.z * b.z;
    result.w = a.w * b.w;
    return result;
}

static inline vec4 vec4_div(vec4 a, vec4 b) {
    vec4 result = {0};
    result.x = a.x / b.x;
    result.y = a.y / b.y;
    result.z = a.z / b.z;
    result.w = a.w / b.w;
    return result;
}

static inline vec4 vec4_negate(vec4 a) {
    vec4 result = {0};
    result.x = -a.x;
    result.y = -a.y;
    result.z = -a.z;
    result.w = -a.w;
    return result;
}

static inline vec4i vec4_equal(vec4 a, vec4 b) {
    vec4i result = {0};
    result.x = a.x == b.x ? 0xffffffff : 0;
    result.y = a.y == b.y ? 0xffffffff : 0;
    result.z = a.z == b.z ? 0xffffffff : 0;
    result.w = a.w == b.w ? 0xffffffff : 0;
    return result;
}

static inline vec4i vec4_cmp_less(vec4 a, vec4 b) {
    vec4i result = {0};
    result.x = a.x < b.x ? 0xffffffff : 0;
    result.y = a.y < b.y ? 0xffffffff : 0;
    result.z = a.z < b.z ? 0xffffffff : 0;
    result.w = a.w < b.w ? 0xffffffff : 0;
    return result;
}

static inline mat4 mat4_diag(float x) {
    mat4 result = {0};
    result.elements[0][0] = x;
    result.elements[1][1] = x;
    result.elements[2][2] = x;
    result.elements[3][3] = x;
    return result;
}

static inline uint16_t vec4_pack_rgb565(vec4 v) {
    uint16_t color = 0;
    v.x = float_clamp(v.x, 0.0, 1.0);
    v.y = float_clamp(v.y, 0.0, 1.0);
    v.z = float_clamp(v.z, 0.0, 1.0);
    color |= ((uint16_t)(v.x * 31.0f)) << 11;
    color |= ((uint16_t)(v.y * 63.0f)) << 5;
    color |= ((uint16_t)(v.z * 31.0f)) << 0;
    return color;
}

static inline vec4 vec4_unpack_rgba8(uint32_t a) {
    return vec4_make(
        (float)((a >>  0) & 0xff) / 255.0f,
        (float)((a >>  8) & 0xff) / 255.0f,
        (float)((a >> 16) & 0xff) / 255.0f,
        (float)((a >> 24) & 0xff) / 255.0f
    );
}

static inline float3 float3_make(float x, float y, float z) {
    float3 result = {0};
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}

static inline float3 rgb565_unpack(uint16_t rgb) {
    float3 result = {0};
    result.x = (float)(rgb >> 11) / 32.0f;
    result.y = (float)(rgb >>  5) / 64.0f;
    result.z = (float)(rgb >>  0) / 32.0f;
    return result;
}

static inline vec4 mat4_mul_vec4(vec4 left, mat4 right) {
    vec4 result = {0};
#if defined(USE_NEON)
    result.neon_f32x4 = vmulq_laneq_f32(right.columns[0].neon_f32x4, left.neon_f32x4, 0);
    result.neon_f32x4 = vfmaq_laneq_f32(result.neon_f32x4, right.columns[1].neon_f32x4, left.neon_f32x4, 1);
    result.neon_f32x4 = vfmaq_laneq_f32(result.neon_f32x4, right.columns[2].neon_f32x4, left.neon_f32x4, 2);
    result.neon_f32x4 = vfmaq_laneq_f32(result.neon_f32x4, right.columns[3].neon_f32x4, left.neon_f32x4, 3);
#else
    result.x = left.elements[0] * right.columns[0].x;
    result.y = left.elements[0] * right.columns[0].y;
    result.z = left.elements[0] * right.columns[0].z;
    result.w = left.elements[0] * right.columns[0].w;

    result.x += left.elements[1] * right.columns[1].x;
    result.y += left.elements[1] * right.columns[1].y;
    result.z += left.elements[1] * right.columns[1].z;
    result.w += left.elements[1] * right.columns[1].w;

    result.x += left.elements[2] * right.columns[2].x;
    result.y += left.elements[2] * right.columns[2].y;
    result.z += left.elements[2] * right.columns[2].z;
    result.w += left.elements[2] * right.columns[2].w;

    result.x += left.elements[3] * right.columns[3].x;
    result.y += left.elements[3] * right.columns[3].y;
    result.z += left.elements[3] * right.columns[3].z;
    result.w += left.elements[3] * right.columns[3].w;
#endif
    return result;
}

static inline mat4 mat4_mul(mat4 left, mat4 right) {
    mat4 result;
    result.columns[0] = mat4_mul_vec4(right.columns[0], left);
    result.columns[1] = mat4_mul_vec4(right.columns[1], left);
    result.columns[2] = mat4_mul_vec4(right.columns[2], left);
    result.columns[3] = mat4_mul_vec4(right.columns[3], left);
    return result;
}

static inline mat4 mat4_translate(vec4 pos) {
    mat4 result = mat4_diag(1.0);
    result.elements[3][0] = pos.x;
    result.elements[3][1] = pos.y;
    result.elements[3][2] = pos.z;
    result.elements[3][3] = 1.0f;
    return result;
}

static inline mat4 mat4_scale(vec4 scale) {
    mat4 result = {0};
    result.elements[0][0] = scale.x;
    result.elements[1][1] = scale.y;
    result.elements[2][2] = scale.z;
    result.elements[3][3] = 1.0f;
    return result;
}

static inline mat4 mat4_rotate(vec4 axis, float angle) {
    mat4 result = {0};

    axis = vec3_normalize(axis);

    float sintheta = sinf(angle);
    float costheta = cosf(angle);
    float cosvalue = 1.0f - costheta;

    result.elements[0][0] = (axis.x * axis.x * cosvalue) + costheta;
    result.elements[0][1] = (axis.x * axis.y * cosvalue) + (axis.z * sintheta);
    result.elements[0][2] = (axis.x * axis.z * cosvalue) - (axis.y * sintheta);

    result.elements[1][0] = (axis.y * axis.x * cosvalue) - (axis.z * sintheta);
    result.elements[1][1] = (axis.y * axis.y * cosvalue) + costheta;
    result.elements[1][2] = (axis.y * axis.z * cosvalue) + (axis.x * sintheta);

    result.elements[2][0] = (axis.z * axis.x * cosvalue) + (axis.y * sintheta);
    result.elements[2][1] = (axis.z * axis.y * cosvalue) - (axis.x * sintheta);
    result.elements[2][2] = (axis.z * axis.z * cosvalue) + costheta;

    result.elements[3][3] = 1.0f;
    return result;
}

static mat4 mat4_perspective(float fov, float aspect, float near, float far) {
    mat4 result = {0};

    // float cotangent = 1.0f / tanf(fov / 2.0f);
    // result.elements[0][0] = cotangent / aspect;
    // result.elements[1][1] = cotangent;
    // result.elements[2][3] = -1.0f;

    // result.elements[2][2] = (far) / (near - far);
    // result.elements[3][2] = (near * far) / (near - far);

    // const float cotangent = 1.0f / tanf(fov * (PI / 360.0f));
    // result.elements[0][0] = cotangent / aspect;
    // result.elements[1][1] = cotangent;
    // result.elements[2][2] = (near + far) / (near - far);
    // result.elements[2][3] = (2.0f * near * far) / (near - far);
    // result.elements[3][2] = -1.0f;


    float fov_rad = fov * (float)PI / 180.0f;
    float f = 1.0f / tanf(fov_rad / 2.0f);

    result.elements[0][0] = f / aspect;
    result.elements[0][1] = 0.0f;
    result.elements[0][2] = 0.0f;
    result.elements[0][3] = 0.0f;

    result.elements[1][0] = 0.0f;
    result.elements[1][1] = f;
    result.elements[1][2] = 0.0f;
    result.elements[1][3] = 0.0f;

    result.elements[2][0] = 0.0f;
    result.elements[2][1] = 0.0f;
    result.elements[2][2] = far / (far - near);
    result.elements[2][3] = 1.0f;

    result.elements[3][0] = 0.0f;
    result.elements[3][1] = 0.0f;
    result.elements[3][2] = -near * far / (far - near);
    result.elements[3][3] = 0.0f;

    return result;
}

static inline mat4 _mat4_lookat(vec4 f,  vec4 s, vec4 u,  vec4 eye) {
    mat4 result;

    result.elements[0][0] = s.x;
    result.elements[0][1] = u.x;
    result.elements[0][2] = -f.x;
    result.elements[0][3] = 0.0f;

    result.elements[1][0] = s.y;
    result.elements[1][1] = u.y;
    result.elements[1][2] = -f.y;
    result.elements[1][3] = 0.0f;

    result.elements[2][0] = s.z;
    result.elements[2][1] = u.z;
    result.elements[2][2] = -f.z;
    result.elements[2][3] = 0.0f;

    result.elements[3][0] = -vec3_dot(s, eye);
    result.elements[3][1] = -vec3_dot(u, eye);
    result.elements[3][2] = vec3_dot(f, eye);
    result.elements[3][3] = 1.0f;

    return result;
}

static mat4 mat4_lookat_rh(vec4 eye, vec4 center, vec4 up) {
    vec4 f = vec3_normalize(vec4_sub(center, eye));
    vec4 s = vec3_normalize(vec3_cross(f, up));
    vec4 u = vec3_cross(s, f);
    return _mat4_lookat(f, s, u, eye);
}

static mat4 mat4_lookat_lh(vec4 eye, vec4 center, vec4 up) {
    vec4 f = vec3_normalize(vec4_sub(eye, center));
    vec4 s = vec3_normalize(vec3_cross(f, up));
    vec4 u = vec3_cross(s, f);
    return _mat4_lookat(f, s, u, eye);
}