import { WebGPU } from "./engine/webgpu";

const wg = new WebGPU;
wg.initialize(<HTMLCanvasElement>document.getElementById('wgpucanvas'))
.then(()=>wg.render());

