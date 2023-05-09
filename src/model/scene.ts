import { Camera } from "./camera";
import { ModelA } from "./modelA";
import { Vec3 } from "../utils/vectors";

export class Scene {
  triangles: ModelA[];
  camera: Camera;

  constructor() {
    this.triangles = [];
    this.triangles.push(new ModelA(Vec3.ZERO(), 10));
    this.triangles.push(new ModelA(new Vec3(0.5,0,-0.5), 15));
    this.camera = new Camera(new Vec3(0,2,2), new Vec3(0.5,0,-0.5), Vec3.J());
  }

  update(time: number, timeDelta: number) {
    this.triangles.forEach(t => { t.update(time,timeDelta) })
    this.camera.update(time, timeDelta);
  }
}