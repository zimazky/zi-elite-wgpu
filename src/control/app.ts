import { initKeyBuffer } from "./keyboard";
import { Scene } from "../model/scene";
import { Engine } from "../view/engine";
import { initGPU } from "../view/webgpu";

export class App {
  canvas: HTMLCanvasElement;
  renderer: Engine | null = null;
  scene: Scene | null = null;
  startTime: number = 0;
  currentTime: number = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    initKeyBuffer();
  }

  async initialize() {
    const {device, context, format} = await initGPU(this.canvas);
    this.renderer = new Engine(device, context, format);
    await this.renderer.initialize();
    this.scene = new Scene();
    this.startTime = this.currentTime = performance.now()/1000.;
  }

  run = () => {
    var running = true;

    const lCurrentTime = performance.now()/1000.;
    const time = lCurrentTime - this.startTime;
    const timeDelta = lCurrentTime - this.currentTime;
    this.currentTime = lCurrentTime;

    if(!this.scene) throw Error('Scene is not defined');
    this.scene.update(time, timeDelta);
    this.renderer?.render(this.scene);

    if(running) {
        requestAnimationFrame(this.run);
    }
  }
}