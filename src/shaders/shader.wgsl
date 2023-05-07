struct Fragment {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> Fragment {
  var positions = array<vec2f, 3> (
    vec2f(0, 0.5),
    vec2f(-0.5, -0.5),
    vec2f(0.5, -0.5)
  );

  var colors = array<vec3f, 3> (
    vec3f(1, 0, 0),
    vec3f(0, 1, 0),
    vec3f(0, 0, 1)
  );

  var output: Fragment;
  output.position = vec4f(positions[vi], 0, 1);
  output.color = vec4f(colors[vi], 1);

  return output;
}

@fragment
fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
  return color;
}