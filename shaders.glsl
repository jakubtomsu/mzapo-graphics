/* quad vertex shader */
@vs vs
in vec2 position;
out vec2 uv;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    uv = position * 0.5 + 0.5;
}
@end

/* quad fragment shader */
@fs fs
in vec2 uv;
out vec4 frag_color;

layout(binding=0) uniform texture2D tex;
layout(binding=0) uniform sampler smp;

void main() {
    frag_color = texture(sampler2D(tex, smp), uv);
}
@end

/* quad shader program */
@program quad vs fs
