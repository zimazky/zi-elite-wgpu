import { App } from "./control/app";

const canvas = <HTMLCanvasElement> document.getElementById('wgpucanvas');
const app = new App(canvas);
app.initialize().then(app.run);

