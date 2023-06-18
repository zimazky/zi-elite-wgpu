export async function initGPU(canvas: HTMLCanvasElement) {
  const gpu = navigator.gpu;
  if(!gpu) throw new Error('Gpu api is not supported on this browser');
  const adapter = await gpu.requestAdapter({
    powerPreference: 'high-performance'
  });
  if(!adapter) throw new Error('Adapter is not defined');
  const device = await adapter.requestDevice({
    requiredFeatures: ['texture-compression-bc'],
    requiredLimits: {
      'maxStorageBufferBindingSize': adapter.limits.maxStorageBufferBindingSize
    }
  });
  console.log(adapter, device);
  if(!device) throw new Error('Device is not defined');
  const context = canvas.getContext('webgpu');
  if(!context) throw new Error('Context is not defined');
  const format = gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: 'opaque'
  });
  return {device, context, format};
}

export async function initPipeline(device: GPUDevice, format: GPUTextureFormat, shaderModule: GPUShaderModule) {
  const pipeline = await device.createRenderPipelineAsync({
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main'
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [{
        format
      }]
    },
    primitive: {
      topology: 'triangle-list'
    },
    layout: 'auto'
  });
  return pipeline;
}
