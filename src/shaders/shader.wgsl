struct TransformData {
  model: mat4x4f,
  view: mat4x4f,
  projection: mat4x4f
};

@group(0) @binding(0) var<uniform> transformUBO: TransformData;
@group(0) @binding(1) var textureA: texture_2d<f32>;
@group(0) @binding(2) var samplerA: sampler;

struct Fragment {
  @builtin(position) position: vec4f,
  @location(0) texcoord: vec2f
};

@vertex
fn vs_main(@location(0) vpos: vec2f, @location(1) vtexcoord: vec2f) -> Fragment {
  var output: Fragment;
  output.position = 
    transformUBO.projection*
    transformUBO.view*
    transformUBO.model*
    vec4f(vpos, 0, 1);
  output.texcoord = vtexcoord;
  return output;
}

@fragment
fn fs_main(fsInput: Fragment) -> @location(0) vec4f {
  return textureSample(textureA, samplerA, fsInput.texcoord);
}