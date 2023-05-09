// ----------------------------------------------------------------------------
// Модуль определения модели треугольника 
// ----------------------------------------------------------------------------

import { Mat4, Vec3, Quaternion } from "../utils/vectors";

export class ModelA {
  position: Vec3;
  /** Период обращения в секундах */
  period: number;
  orientation: Quaternion = Quaternion.Identity();
  transform: Mat4 = Mat4.ID();

  constructor(pos: Vec3, period: number) {
    this.position = pos.copy();
    this.period = period;
  }

  update(time: number, timeDelta: number) {
    this.orientation = Quaternion.fromAxisAngle(Vec3.J(), 2.*Math.PI*(time/this.period)).normalize() as Quaternion;
    this.transform = Mat4.modelFromQuatPosition(this.orientation, this.position);
  }

}