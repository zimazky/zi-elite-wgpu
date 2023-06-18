import { Scene } from 'src/model/scene';
import { loadImg } from '../utils/loadimg';
import { Material } from './material';
import { Triangle } from './triangle';
import shader from '/src/shaders/shader.wgsl';

const uniformOffset = 256;  // выравнивание буфера для разных биндинг-групп

class Program {
  pipeline: GPURenderPipeline;
  uniformBuffer: GPUBuffer;
  bindGroups: GPUBindGroup[] = []; // группы для разных моделей

  constructor(pipeline: GPURenderPipeline, uniformBuffer: GPUBuffer) {
    this.pipeline = pipeline;
    this.uniformBuffer = uniformBuffer;
  }
}

export class Engine {
  format: GPUTextureFormat;
  device: GPUDevice;
  context: GPUCanvasContext;

  uniformBuffer: GPUBuffer;
  pipeline: GPURenderPipeline;
  bindGroups: GPUBindGroup[] = []; // группы для разных треугольников
  triangle: Triangle;
  material: Material;

  programs: Program[] = [];

  constructor(device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat = 'bgra8unorm') {
    this.device = device;
    this.context = context;
    this.format = format;
  }

  addProgram(fn: (e: Engine)=>{pipeline: GPURenderPipeline, uniformBuffer: GPUBuffer}) {
    const {pipeline, uniformBuffer} = fn(this);
    this.programs.push(new Program(pipeline, uniformBuffer));
  }

  async initialize() {
    this.triangle = new Triangle(this.device);
    this.material = new Material();
    const img = await loadImg('textures/cat-head.jpg');
    this.material.initialize(this.device,img);

    const uniformSize = 3*4*16; // две матрицы преобразования
    const uniformBufferSize = uniformOffset + uniformSize; // общий размер буфера

    this.uniformBuffer = this.device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {}},
        {binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {}},
        {binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {}}
      ]
    });
    // группа для первого треугольника
    this.bindGroups[0] = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {binding: 0, resource: {
          buffer: this.uniformBuffer,
          offset: 0,
          size: uniformSize
        }},
        {binding: 1, resource: this.material.view},
        {binding: 2, resource: this.material.sampler}
      ]
    });
    // группа для второго треугольника
    this.bindGroups[1] = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {binding: 0, resource: {
          buffer: this.uniformBuffer,
          offset: uniformOffset,
          size: uniformSize
        }},
        {binding: 1, resource: this.material.view},
        {binding: 2, resource: this.material.sampler}
      ]
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
  }

  render = (scene: Scene)=>{
    const projection = scene.camera.projection.copy();
    //const view = Mat4.lookAt(new Vec3(0,2,2), new Vec3(0.5,0,0), Vec3.J()) //camera.transform;
    const view = scene.camera.transform.copy();

    const textureView = this.context.getCurrentTexture().createView();
    const commandEncoder = this.device.createCommandEncoder();
    const renderpass = commandEncoder.beginRenderPass({
      label: 'basic canvas renderpass',
      colorAttachments: [{
        view: textureView,
        clearValue: {r: 0.5, g: 0, b: 0.25, a: 1},
        loadOp: 'clear',
        storeOp: 'store'
      }]
    });
    renderpass.setPipeline(this.pipeline);
    renderpass.setVertexBuffer(0, this.triangle.buffer);

    scene.triangles.forEach((t,i)=>{
      const model = t.transform.copy();
      this.device.queue.writeBuffer(this.uniformBuffer, i*uniformOffset, new Float32Array(model.getArray()));
      this.device.queue.writeBuffer(this.uniformBuffer, i*uniformOffset + 64, new Float32Array(view.getArray()));
      this.device.queue.writeBuffer(this.uniformBuffer, i*uniformOffset + 128, new Float32Array(projection.getArray()));
      renderpass.setBindGroup(0, this.bindGroups[i]);
      renderpass.draw(3, 1, 0, 0);
      //console.log(t);
    })
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
