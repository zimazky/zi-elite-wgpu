import { rad } from './mathutils';
import { Triangle } from './triangle';
import { Mat4, Vec3 } from './vectors';
import shader from '/src/shaders/shader.wgsl';

export class WebGPU {
  private format: GPUTextureFormat = 'bgra8unorm';
  private gpu: GPU;
  private adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;

  uniformBuffer: GPUBuffer;
  pipeline: GPURenderPipeline;
  bindGroup: GPUBindGroup;
  renderPassDescriptor: GPURenderPassDescriptor;
  triangle: Triangle;

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

    this.triangle = new Triangle(this.device);

    this.uniformBuffer = this.device.createBuffer({
      size: 64*3,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [{binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {}}]
    });
    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{binding: 0, resource: {buffer: this.uniformBuffer}}]
    });
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
  
    const shaderModule = this.device.createShaderModule({code: shader});
    this.pipeline = this.device.createRenderPipeline({
      label: 'triangle shaider',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [this.triangle.bufferLayout]
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

    this.renderPassDescriptor = {
      label: 'basic canvas renderpass',
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: {r: 0.5, g: 0, b: 0.25, a: 1},
        loadOp: 'clear',
        storeOp: 'store'
      }]
    } as GPURenderPassDescriptor;
  }
  
  render() {
    const projection = Mat4.perspectiveDx(rad(45), 1, 0.01, 10);
    const view = Mat4.lookAt(new Vec3(0.0,6.,-2.5), new Vec3(-0.,0.,-1.), Vec3.J());
    const model = Mat4.rotateMat(Vec3.J(), rad(20));

    this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array(model.getArray()));
    this.device.queue.writeBuffer(this.uniformBuffer, 64, new Float32Array(view.getArray()));
    this.device.queue.writeBuffer(this.uniformBuffer, 128, new Float32Array(projection.getArray()));

    const commandEncoder = this.device.createCommandEncoder();
    //this.renderPassDescriptor.colorAttachments[0].view = this.context.getCurrentTexture().createView();
    const renderpass = commandEncoder.beginRenderPass(this.renderPassDescriptor);
    renderpass.setPipeline(this.pipeline);
    renderpass.setBindGroup(0, this.bindGroup);
    renderpass.setVertexBuffer(0, this.triangle.buffer);
    renderpass.draw(3, 1, 0, 0);
    renderpass.end();
  
    this.device.queue.submit([commandEncoder.finish()]);
  }
/*
  const observer = new ResizeObserver(entries => {
    for (const entry of entries) {
      if(entry.target.id)
      const canvas = entry.target;
      const width = entry.contentBoxSize[0].inlineSize;
      const height = entry.contentBoxSize[0].blockSize;
      canvas.width = Math.min(width, device.limits.maxTextureDimension2D);
      canvas.height = Math.min(height, device.limits.maxTextureDimension2D);
      // re-render
      render();
    }
  });
  observer.observe(canvas);
  */

}
