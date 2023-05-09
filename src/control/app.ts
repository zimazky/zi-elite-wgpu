import { initKeyBuffer } from "../core/keyboard";
import { Scene } from "../model/scene";
import { WebGPU } from "../view/webgpu";

export class App {
  canvas: HTMLCanvasElement;
  renderer: WebGPU;
  scene: Scene;
  startTime: number = 0;
  currentTime: number = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.renderer = new WebGPU();
    this.scene = new Scene();
    initKeyBuffer();
  }

  async initialize() {
    await this.renderer.initialize(this.canvas);
    this.startTime = this.currentTime = performance.now()/1000.;
  }

  run = () => {
    var running = true;

    const lCurrentTime = performance.now()/1000.;
    const time = lCurrentTime - this.startTime;
    const timeDelta = lCurrentTime - this.currentTime;
    this.currentTime = lCurrentTime;

    this.scene.update(time, timeDelta);
    this.renderer.render(this.scene.camera, this.scene.triangles);

    if(running) {
        requestAnimationFrame(this.run);
    }
  }
}