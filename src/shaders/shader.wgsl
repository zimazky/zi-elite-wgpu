struct TransformData {
  model: mat4x4f,
  view: mat4x4f,
  projection: mat4x4f
};

@group(0) @binding(0) var<uniform> transformUBO: TransformData;

struct Fragment {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f
};

@vertex
fn vs_main(@location(0) vpos: vec2f, @location(1) vcolor: vec3f) -> Fragment {
  var output: Fragment;
  output.position = 
    transformUBO.projection*
    transformUBO.view*
    transformUBO.model*
    vec4f(vpos, 0, 1);
  output.color = vec4f(vcolor, 1);
  return output;
}

@fragment
fn fs_main(fsInput: Fragment) -> @location(0) vec4f {
  return fsInput.color;
}