export class Triangle {
  buffer: GPUBuffer;
  bufferLayout: GPUVertexBufferLayout;

  constructor(device: GPUDevice) {
    // x y r g b
    const vertices = new Float32Array([
       0.0,  0.5, 1, 0, 0,
      -0.5, -0.5, 0, 1, 0,
       0.5, -0.5, 0, 0, 1
    ]);
    const descriptor: GPUBufferDescriptor = {
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    }
    this.buffer = device.createBuffer(descriptor);
    new Float32Array(this.buffer.getMappedRange()).set(vertices);
    this.buffer.unmap();
    this.bufferLayout = {
      arrayStride: 20,
      attributes: [
        {shaderLocation: 0, format: 'float32x2', offset: 0},
        {shaderLocation: 1, format: 'float32x3', offset: 8}
      ]
    };
  }
}