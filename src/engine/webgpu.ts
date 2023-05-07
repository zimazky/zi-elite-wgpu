import shader from '/src/shaders/shader.wgsl';

export class WebGPU {
  gpu: GPU | null = null;
  adapter: GPUAdapter | null = null;
  device?: GPUDevice;
  format: GPUTextureFormat = 'bgra8unorm';
  context: GPUCanvasContext | null = null;

  async initialize(canvas: HTMLCanvasElement) {
    this.gpu = navigator.gpu;
    this.adapter = await this.gpu?.requestAdapter();
    this.device = await this.adapter?.requestDevice();
    if(!this.device) throw new Error('WebGPU is not supported on this browser');
    this.context = canvas.getContext('webgpu');
    if(!this.context) throw new Error('Context is not defined');
    this.format = this.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque'
    })
  }

  start() {
    if(!this.device) throw new Error('Device is not defined');
    if(!this.context) throw new Error('Context is not defined');

    const bindGroupLayout = this.device.createBindGroupLayout({entries: []});
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: []
    });
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
  
    const shaderModule = this.device.createShaderModule({code: shader});
    const pipeline = this.device.createRenderPipeline({
      label: 'triangle shaider',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main'
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{format: this.format}]
      },
      primitive: {
        topology: 'triangle-list'
      },
      layout: pipelineLayout
    });
  
    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();
    const renderpass = commandEncoder.beginRenderPass({
      label: 'basic canvas renderpass',
      colorAttachments: [{
        view: textureView,
        clearValue: {r: 0.5, g: 0, b: 0.25, a: 1},
        loadOp: 'clear',
        storeOp: 'store'
      }]
    })
    renderpass.setPipeline(pipeline);
    renderpass.setBindGroup(0, bindGroup);
    renderpass.draw(3, 1, 0, 0);
    renderpass.end();
  
    this.device.queue.submit([commandEncoder.finish()]);
  }
}
