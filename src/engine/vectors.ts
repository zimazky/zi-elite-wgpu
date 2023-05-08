const EPSILON = 0.00001;

interface IVectors<T> {
  /** Проверка на равенство в пределах точности EPSILON */
  equals(v: T): boolean;
  /** Мутабельное сложение с вектором */
  addMutable(v: T): T;
  /** Мутабельное вычитание вектора */
  subMutable(v: T): T;
  /** Мутабельное произведение со скаляром */
  mulMutable(f: number): T;
  /** Мутабельное деление на скаляр */
  divMutable(f: number): T;
  /** Мутабельное поэлементное произведение векторов */
  mulElMutable(v: T): T;
  /** Мутабельное поэлементное деление */
  divElMutable(v: T): T;
  /** Мутабельная операция нормализации вектора */
  normalizeMutable(): T;
  /** Иммутабельное сложение с вектором */
  add(v: T): T;
  /** Иммутабельное вычитание вектора */
  sub(v: T): T;
  /** Иммутабельное произведение со скаляром */
  mul(f: number): T;
  /** Иммутабельное деление на скаляр */
  div(f: number): T;
  /** Иммутабельное поэлементное произведение векторов */
  mulEl(v: T): T;
  /** Иммутабельное поэлементное деление */
  divEl(v: T): T;
  /** Иммутабельная операция нормализации вектора */
  normalize(): T;
  /** Длина вектора */
  length(): number;
  /** Скалярное произведение векторов */
  dot(v: T): number;
  /** Копия вектора */
  copy(): T;
  /** Вектор с целыми частями элементов */
  floor(): T;
  /** Вектор с дробными частями элементов */
  fract(): T;
  /** Получить компоненты в виде массива */
  getArray(): number[];
}

interface IMatrixes<TMat, TVec> {
  /** Проверка на равенство в пределах точности EPSILON */
  equals(v: TMat): boolean;
  /** Копия матрицы */
  copy(): TMat;
  /** Транспонированная матрица */
  transpose(): TMat;
  /** Мутабельное сложение матриц */
  addMatMutable(m: TMat): TMat;
  /** Иммутабельное сложение матриц */
  addMat(m: TMat): TMat;
  /** Мутабельное вычитание матрицы */
  subMatMutable(m: TMat): TMat;
  /** Иммутабельное вычитание матрицы */
  subMat(m: TMat): TMat;
  /** Мутабельное умножение на скаляр */
  mulMutable(n: number): TMat;
  /** Иммутабельное умножение на скаляр */
  mul(n: number): TMat;
  /** Мутабельное деление на скаляр */
  divMutable(n: number): TMat;
  /** Иммутабельное деление на скаляр */
  div(n: number): TMat;

  /** Иммутабельное произведение матриц */
  mulMat(m: TMat): TMat;
  /** Иммутабельное произведение матриц слева */
  mulMatLeft(m: TMat): TMat;
  /** Иммутабельное произведение матрицы на вектор справа */
  mulVec(v: TVec): TVec;
  /** 
   * Иммутабельное произведение транспонированного вектора на матрицу
   * (произведение матрицы на вектор слева) 
   * */
  mulVecLeft(v: TVec): TVec;
  /** Получить компоненты в виде массива */
  getArray(): number[];
}

/****************************************************************************** 
 * Класс четырехмерного вектора 
 * */
export class Vec4 implements IVectors<Vec4> {
  x: number; y: number; z: number; w: number;
  static ZERO = () => new Vec4(0.,0.,0.,0.);
  static ONE = () => new Vec4(1.,1.,1.,1.);
  static I = () => new Vec4(1.,0.,0.,0.);
  static J = () => new Vec4(0.,1.,0.,0.);
  static K = () => new Vec4(0.,0.,1.,0.);
  static L = () => new Vec4(0.,0.,0.,1.);
  static RAND = () => new Vec4(Math.random(),Math.random(),Math.random(),Math.random());

  constructor(x: number, y: number, z: number, w: number) {
    this.x = x; this.y = y; this.z = z; this.w = w;
  }

  toString(): string {
    return `(${this.x.toFixed(2)}, ${this.y.toFixed(2)}, ${this.z.toFixed(2)}, ${this.w.toFixed(2)})`;
  }

  equals(v: Vec4): boolean {
    return Math.abs(this.x-v.x)<=EPSILON*Math.max(1., this.x, v.x)
    && Math.abs(this.y-v.y)<=EPSILON*Math.max(1., this.y, v.y)
    && Math.abs(this.z-v.z)<=EPSILON*Math.max(1., this.z, v.z)
    && Math.abs(this.w-v.w)<=EPSILON*Math.max(1., this.w, v.w);
  }

  addMutable(v: Vec4): Vec4 {
    this.x += v.x; this.y += v.y; this.z += v.z; this.w += v.w;
    return this;
  }

  subMutable(v: Vec4): Vec4 {
    this.x -= v.x; this.y -= v.y; this.z -= v.z; this.w -= v.w;
    return this;
  }

  mulMutable(f: number): Vec4 {
    this.x *= f; this.y *= f; this.z *= f; this.w *= f;
    return this;
  }

  divMutable(f: number): Vec4 {
    this.x /= f; this.y /= f; this.z /= f; this.w /= f;
    return this;
  }

  mulElMutable(v: Vec4): Vec4 {
    this.x *= v.x; this.y *= v.y; this.z *= v.z; this.w *= v.w;
    return this;
  }

  divElMutable(v: Vec4): Vec4 {
    this.x /= v.x; this.y /= v.y; this.z /= v.z; this.w /= v.w;
    return this;
  }

  normalizeMutable(): Vec4 { return this.divMutable(Math.sqrt(this.dot(this))); }

  add(v: Vec4): Vec4 { return this.copy().addMutable(v); }

  sub(v: Vec4): Vec4 { return this.copy().subMutable(v); }

  mul(f: number): Vec4 { return this.copy().mulMutable(f); }

  div(f: number): Vec4 { return this.copy().divMutable(f); }

  mulEl(v: Vec4): Vec4 { return this.copy().mulElMutable(v); }

  divEl(v: Vec4): Vec4 { return this.copy().divElMutable(v); }

  normalize(): Vec4 { return this.copy().normalizeMutable(); }
  
  length(): number { return Math.sqrt(this.dot(this)); }

  dot(v: Vec4): number { return this.x*v.x + this.y*v.y + this.z*v.z + this.w*v.w; }

  copy(): Vec4 { return new Vec4(this.x, this.y, this.z, this.w); }

  floor(): Vec4 {
    return new Vec4(Math.floor(this.x), Math.floor(this.y), Math.floor(this.z), Math.floor(this.w));
  }

  fract(): Vec4 { return this.copy().subMutable(this.floor()); }

  getArray(): number[] { return [this.x, this.y, this.z, this.w]; }
}

/****************************************************************************** 
 * Класс четырехмерной матрицы 
 * */
export class Mat4 implements IMatrixes<Mat4, Vec4> {
  i: Vec4; j: Vec4; k: Vec4; l: Vec4;
  static ID = () => new Mat4(Vec4.I(), Vec4.J(), Vec4.K(), Vec4.L());
  static ZERO = () => new Mat4(Vec4.ZERO(), Vec4.ZERO(), Vec4.ZERO(), Vec4.ZERO());
  static RAND = () => new Mat4(Vec4.RAND(), Vec4.RAND(), Vec4.RAND(), Vec4.RAND());

  constructor(i: Vec4, j: Vec4, k: Vec4, l: Vec4) {
    this.i = i.copy(); this.j = j.copy(); this.k = k.copy(); this.l = l.copy();
  }

  cx(): Vec4 { return new Vec4(this.i.x, this.j.x, this.k.x, this.l.x); }
  cy(): Vec4 { return new Vec4(this.i.y, this.j.y, this.k.y, this.l.y); }
  cz(): Vec4 { return new Vec4(this.i.z, this.j.z, this.k.z, this.l.z); }
  cw(): Vec4 { return new Vec4(this.i.w, this.j.w, this.k.w, this.l.w); }

  equals(v: Mat4): boolean {
    return this.i.equals(v.i) && this.j.equals(v.j) && this.k.equals(v.k) && this.l.equals(v.l);
  }

  copy(): Mat4 {
    return new Mat4(this.i.copy(), this.j.copy(), this.k.copy(), this.l.copy());  
  }

  transpose(): Mat4 { return new Mat4(this.cx(), this.cy(), this.cz(), this.cw()); }

  addMatMutable(m: Mat4): Mat4 {
    this.i.addMutable(m.i);
    this.j.addMutable(m.j);
    this.k.addMutable(m.k);
    this.l.addMutable(m.l);
    return this;
  }

  subMatMutable(m: Mat4): Mat4 {
    this.i.subMutable(m.i);
    this.j.subMutable(m.j);
    this.k.subMutable(m.k);
    this.l.subMutable(m.l);
    return this;
  }

  mulMutable(n: number): Mat4 {
    this.i.mulMutable(n);
    this.j.mulMutable(n);
    this.k.mulMutable(n);
    this.l.mulMutable(n);
    return this;
  }

  divMutable(n: number): Mat4 {
    this.i.divMutable(n);
    this.j.divMutable(n);
    this.k.divMutable(n);
    this.l.divMutable(n);
    return this;
  }

  addMat(m: Mat4): Mat4 { return this.copy().addMatMutable(m); }

  subMat(m: Mat4): Mat4 { return this.copy().subMatMutable(m); }

  mul(n: number): Mat4 { return this.copy().mulMutable(n); }

  div(n: number): Mat4 { return this.copy().divMutable(n); }

  mulMat(m: Mat4): Mat4 {
    return new Mat4(
      new Vec4(this.cx().dot(m.i), this.cy().dot(m.i), this.cz().dot(m.i), this.cw().dot(m.i)),
      new Vec4(this.cx().dot(m.j), this.cy().dot(m.j), this.cz().dot(m.j), this.cw().dot(m.j)),
      new Vec4(this.cx().dot(m.k), this.cy().dot(m.k), this.cz().dot(m.k), this.cw().dot(m.k)),
      new Vec4(this.cx().dot(m.l), this.cy().dot(m.l), this.cz().dot(m.l), this.cw().dot(m.l))
    );
  }

  mulMatLeft(m: Mat4): Mat4 {
    return new Mat4(
      new Vec4(this.i.dot(m.cx()), this.i.dot(m.cy()), this.i.dot(m.cz()), this.i.dot(m.cw())),
      new Vec4(this.j.dot(m.cx()), this.j.dot(m.cy()), this.j.dot(m.cz()), this.j.dot(m.cw())),
      new Vec4(this.k.dot(m.cx()), this.k.dot(m.cy()), this.k.dot(m.cz()), this.k.dot(m.cw())),
      new Vec4(this.l.dot(m.cx()), this.l.dot(m.cy()), this.l.dot(m.cz()), this.l.dot(m.cw()))
    );
  }

  mulVec(v: Vec4): Vec4 {
    return new Vec4(this.cx().dot(v), this.cy().dot(v), this.cz().dot(v), this.cw().dot(v));
  }

  mulVecLeft(v: Vec4): Vec4 { return new Vec4(this.i.dot(v), this.j.dot(v), this.k.dot(v), this.l.dot(v)); }

  getArray(): number[] {
    return [
      this.i.x, this.i.y, this.i.z, this.i.w,
      this.j.x, this.j.y, this.j.z, this.j.w,
      this.k.x, this.k.y, this.k.z, this.k.w,
      this.l.x, this.l.y, this.l.z, this.l.w,
    ]
  }

  /**
   * Получить матрицу ортогональной проекции OpenGL/WebGL
   * соответствующую области отсечения по z в интервале -1...1
   * @param left - расстояние до левой плоскости отсечения
   * @param right - расстояние до правой плоскости отсечения
   * @param bottom - расстояние до нижней плоскости отсечения
   * @param top - расстояние до верхней плоскости отсечения
   * @param near - расстояние до ближней плоскости отсечения
   * @param far - расстояние до дальней плоскости отсечения
   * @returns матрица ортогональной проекции
   */
  static orthoGl(
    left: number, right: number, bottom: number, top: number, near: number, far: number): Mat4 {
    const lr = 1. / (left - right);
    const bt = 1. / (bottom - top);
    const nf = 1. / (near - far);
    return new Mat4(
      new Vec4(-2.*lr, 0., 0., 0.),
      new Vec4(0., -2.*bt, 0., 0.),
      new Vec4(0., 0., 2.*nf,  0.),
      new Vec4((left+right)*lr, (top+bottom)*bt, (far+near)*nf, 1.)
    );
  }
  /**
   * Получить матрицу ортогональной проекции DirectX/WebGPU/Vulkan/Metal
   * соответствующую области отсечения по z в интервале 0...1
   * @param left - расстояние до левой плоскости отсечения
   * @param right - расстояние до правой плоскости отсечения
   * @param bottom - расстояние до нижней плоскости отсечения
   * @param top - расстояние до верхней плоскости отсечения
   * @param near - расстояние до ближней плоскости отсечения
   * @param far - расстояние до дальней плоскости отсечения
   * @returns матрица ортогональной проекции
   */
  static orthoDx(
    left: number, right: number, bottom: number, top: number, near: number, far: number): Mat4 {
    const lr = 1. / (left - right);
    const bt = 1. / (bottom - top);
    const nf = 1. / (near - far);
    return new Mat4(
      new Vec4(-2.*lr, 0., 0., 0.),
      new Vec4(0., -2.*bt, 0., 0.),
      new Vec4(0., 0., nf,  0.),
      new Vec4((left+right)*lr, (top+bottom)*bt, near*nf, 1.)
    );
  }
  /**
   * Получить матрицу перспективной проекции OpenGL/WebGL
   * соответствующую области отсечения по z в интервале -1...1
   * @param fovy - величина угла поля зрения по вертикали в радианах
   * @param aspect - соотношение сторон (ширина/высота)
   * @param near - расстояние до ближней плоскости отсечения, должно быть больше 0
   * @param far - расстояние до дальней плоскости отсечения, должно быть больше 0
   * @returns матрица перспективной проекции
   */
  static perspectiveGl(fovy: number, aspect: number, near: number, far: number): Mat4 {
    const f = 1./Math.tan(0.5*fovy);
    return new Mat4(
      new Vec4(f/aspect, 0., 0., 0.),
      new Vec4(0., f, 0., 0.),
      new Vec4(0., 0., -(far+near)/(far-near), -1.),
      new Vec4(0., 0., -2.*far*near/(far-near), 0.)
    );
  }
  /**
   * Получить матрицу перспективной проекции DirectX/WebGPU/Vulkan/Metal
   * соответствующую области отсечения по z в интервале 0...1
   * @param fovy - величина угла поля зрения по горизонтали в радианах
   * @param aspect - соотношение сторон (ширина/высота)
   * @param near - расстояние до ближней плоскости отсечения, должно быть больше 0
   * @param far - расстояние до дальней плоскости отсечения, должно быть больше 0
   * @returns матрица перспективной проекции
   */
  static perspectiveDx(fovy: number, aspect: number, near: number, far: number): Mat4 {
    const f = 1./Math.tan(0.5*fovy);
    let e10 = -1;
    let e14 = -near;
    if(far != null && far !== Infinity) {
      const nf = 1 / (near - far);
      e10 = far * nf;
      e14 = far * near * nf;
    }
    return new Mat4(
      new Vec4(f/aspect, 0., 0., 0.),
      new Vec4(0., f, 0., 0.),
      new Vec4(0., 0., e10, -1.),
      new Vec4(0., 0., e14, 0.)
    );
  }
  /**
   * Получить матрицу вида
   * @param from - точка наблюдения
   * @param to - точка направления взгляда
   * @param up - направление вверх
   * @returns 
   */
  static lookAt(from: Vec3, to: Vec3, up: Vec3): Mat4 { 
    const forward = from.sub(to).normalize();
    const right = up.cross(forward).normalize();
    const newup = forward.cross(right);
    return new Mat4(
      new Vec4(right.x, newup.x, forward.x, 0),
      new Vec4(right.y, newup.y, forward.y, 0),
      new Vec4(right.z, newup.z, forward.z, 0),
      new Vec4(-right.dot(from), -newup.dot(from), -forward.dot(from), 1.)
    );
  } 

  /**
   * Получить матрицу вращения
   * @param axis - ось вращения
   * @param theta - угол поворота
   * @returns 
   */
  static rotateMat(axis: Vec3, theta: number): Mat4 {
    const a = axis.normalize();
    const c = Math.cos(theta);
    const mc = 1 - c;
    const s = Math.sin(theta);
    return new Mat4(
      new Vec4(a.x*a.x*mc+c, a.x*a.y*mc+a.z*s, a.x*a.z*mc-a.y*s, 0),
      new Vec4(a.y*a.x*mc-a.z*s, a.y*a.y*mc+c, a.y*a.z*mc+a.x*s, 0),
      new Vec4(a.z*a.x*mc+a.y*s, a.z*a.y*mc-a.x*s, a.z*a.z*mc+c, 0),
      new Vec4(0, 0, 0, 1)
    );
  } 
}

/****************************************************************************** 
 * Класс трехмерного вектора 
 * */
export class Vec3 implements IVectors<Vec3> {
  x: number; y: number; z: number;
  static ZERO = () => new Vec3(0.,0.,0.);
  static ONE = () => new Vec3(1.,1.,1.);
  static I = () => new Vec3(1.,0.,0.);
  static J = () => new Vec3(0.,1.,0.);
  static K = () => new Vec3(0.,0.,1.);
  static RAND = () => new Vec3(Math.random(),Math.random(),Math.random());

  constructor(x: number, y: number, z: number) {
    this.x = x; this.y = y; this.z = z;
  }

  toString(): string {
    return `(${this.x.toFixed(2)}, ${this.y.toFixed(2)}, ${this.z.toFixed(2)})`;
  }

  equals(v: Vec3): boolean {
    return Math.abs(this.x-v.x)<=EPSILON*Math.max(1., this.x, v.x)
    && Math.abs(this.y-v.y)<=EPSILON*Math.max(1., this.y, v.y)
    && Math.abs(this.z-v.z)<=EPSILON*Math.max(1., this.z, v.z)
  }

  addMutable(v: Vec3): Vec3 {
    this.x += v.x; this.y += v.y; this.z += v.z;
    return this;
  }

  subMutable(v: Vec3): Vec3 {
    this.x -= v.x; this.y -= v.y; this.z -= v.z;
    return this;
  }

  mulMutable(f: number): Vec3 {
    this.x *= f; this.y *= f; this.z *= f;
    return this;
  }

  divMutable(f: number): Vec3 {
    this.x /= f; this.y /= f; this.z /= f;
    return this;
  }

  mulElMutable(v: Vec3): Vec3 {
    this.x *= v.x; this.y *= v.y; this.z *= v.z;
    return this;
  }

  divElMutable(v: Vec3): Vec3 {
    this.x /= v.x; this.y /= v.y; this.z /= v.z;
    return this;
  }

  normalizeMutable(): Vec3 { return this.divMutable(Math.sqrt(this.dot(this))); }

  safeNormalize(): Vec3 { 
    const d = Math.sqrt(this.dot(this));
    if(d<Number.MIN_VALUE) return Vec3.ZERO();
    return this.div(d); 
  }

  colorNormalize(): Vec3 {
    const max = Math.max(this.x, this.y, this.z);
    if(max<Number.MIN_VALUE) return Vec3.ZERO();
    return this.div(max);
  }

  add(v: Vec3): Vec3 { return this.copy().addMutable(v); }

  sub(v: Vec3): Vec3 { return this.copy().subMutable(v); }

  mul(f: number): Vec3 { return this.copy().mulMutable(f); }

  div(f: number): Vec3 { return this.copy().divMutable(f); }

  mulEl(v: Vec3): Vec3 { return this.copy().mulElMutable(v); }

  divEl(v: Vec3): Vec3 { return this.copy().divElMutable(v); }

  normalize(): Vec3 { return this.copy().normalizeMutable(); }

  length(): number { return Math.sqrt(this.dot(this)); }

  dot(v: Vec3): number { return this.x*v.x + this.y*v.y + this.z*v.z; }

  copy(): Vec3 { return new Vec3(this.x, this.y, this.z); }

  floor(): Vec3 { return new Vec3(Math.floor(this.x), Math.floor(this.y), Math.floor(this.z)); }

  fract(): Vec3 { return this.copy().subMutable(this.floor()); }

  exp(): Vec3 {
    return new Vec3(Math.exp(this.x), Math.exp(this.y), Math.exp(this.z));
  }

  getArray(): number[] { return [this.x, this.y, this.z]; }

/** Векторное произведение */
  cross(v: Vec3): Vec3 {
    return new Vec3(
      this.y*v.z - this.z*v.y,
      this.z*v.x - this.x*v.z,
      this.x*v.y - this.y*v.x
    );
  }
}

/****************************************************************************** 
 * Класс трехмерной матрицы 
 * */
export class Mat3 implements IMatrixes<Mat3, Vec3> {
  i: Vec3; j: Vec3; k: Vec3;
  static ID = () => new Mat3(Vec3.I(), Vec3.J(), Vec3.K());
  static ZERO = () => new Mat3(Vec3.ZERO(), Vec3.ZERO(), Vec3.ZERO());

  cx(): Vec3 { return new Vec3(this.i.x, this.j.x, this.k.x); }
  cy(): Vec3 { return new Vec3(this.i.y, this.j.y, this.k.y); }
  cz(): Vec3 { return new Vec3(this.i.z, this.j.z, this.k.z); }

  constructor(i: Vec3, j: Vec3, k: Vec3) {
    this.i = i.copy(); this.j = j.copy(); this.k = k.copy();
  }

  equals(v: Mat3): boolean {
    return this.i.equals(v.i) && this.j.equals(v.j) && this.k.equals(v.k);
  }

  copy(): Mat3 {
    return new Mat3(this.i.copy(), this.j.copy(), this.k.copy());  
  }

  transpose(): Mat3 { return new Mat3(this.cx(), this.cy(), this.cz()); }

  addMatMutable(m: Mat3): Mat3 {
    this.i.addMutable(m.i);
    this.j.addMutable(m.j);
    this.k.addMutable(m.k);
    return this;
  }

  subMatMutable(m: Mat3): Mat3 {
    this.i.subMutable(m.i);
    this.j.subMutable(m.j);
    this.k.subMutable(m.k);
    return this;
  }

  mulMutable(n: number): Mat3 {
    this.i.mulMutable(n);
    this.j.mulMutable(n);
    this.k.mulMutable(n);
    return this;
  }

  divMutable(n: number): Mat3 {
    this.i.divMutable(n);
    this.j.divMutable(n);
    this.k.divMutable(n);
    return this;
  }

  addMat(m: Mat3): Mat3 { return this.copy().addMatMutable(m); }

  subMat(m: Mat3): Mat3 { return this.copy().subMatMutable(m); }

  mul(n: number): Mat3 { return this.copy().mulMutable(n); }

  div(n: number): Mat3 { return this.copy().divMutable(n); }

  mulMat(m: Mat3): Mat3 {
    return new Mat3(
      new Vec3(this.cx().dot(m.i), this.cy().dot(m.i), this.cz().dot(m.i)),
      new Vec3(this.cx().dot(m.j), this.cy().dot(m.j), this.cz().dot(m.j)),
      new Vec3(this.cx().dot(m.k), this.cy().dot(m.k), this.cz().dot(m.k))
    );
  }

  mulMatLeft(m: Mat3): Mat3 {
    return new Mat3(
      new Vec3(this.i.dot(m.cx()), this.i.dot(m.cy()), this.i.dot(m.cz())),
      new Vec3(this.j.dot(m.cx()), this.j.dot(m.cy()), this.j.dot(m.cz())),
      new Vec3(this.k.dot(m.cx()), this.k.dot(m.cy()), this.k.dot(m.cz()))
    );
  }

  mulVec(v: Vec3): Vec3 {
    return new Vec3(this.cx().dot(v), this.cy().dot(v), this.cz().dot(v));
  }

  mulVecLeft(v: Vec3): Vec3 { return new Vec3(this.i.dot(v), this.j.dot(v), this.k.dot(v)); }

  getArray(): number[] {
    return [
      this.i.x, this.i.y, this.i.z,
      this.j.x, this.j.y, this.j.z,
      this.k.x, this.k.y, this.k.z
    ]
  }

  /**
   * Получить матрицу вращения
   * @param axis - ось вращения
   * @param theta - угол поворота
   * @returns 
   */
  static rotateMat(axis: Vec3, theta: number): Mat3 {
    const a = axis.normalize();
    const c = Math.cos(theta);
    const mc = 1 - c;
    const s = Math.sin(theta);
    return new Mat3(
      new Vec3(a.x*a.x*mc+c, a.x*a.y*mc+a.z*s, a.x*a.z*mc-a.y*s),
      new Vec3(a.y*a.x*mc-a.z*s, a.y*a.y*mc+c, a.y*a.z*mc+a.x*s),
      new Vec3(a.z*a.x*mc+a.y*s, a.z*a.y*mc-a.x*s, a.z*a.z*mc+c)
    );
  } 
}

/****************************************************************************** 
 * Класс двумерного вектора 
 * */
export class Vec2 implements IVectors<Vec2> {
  x: number; y: number;
  static ZERO = () => new Vec2(0.,0.);
  static ONE = () => new Vec2(1.,1.);
  static I = () => new Vec2(1.,0.);
  static J = () => new Vec2(0.,1.);
  static RAND = () => new Vec2(Math.random(),Math.random());

  constructor(x: number, y: number) {
    this.x = x; this.y = y;
  }

  toString(): string {
    return `(${this.x.toFixed(2)}, ${this.y.toFixed(2)})`;
  }

  equals(v: Vec2): boolean {
    return Math.abs(this.x-v.x)<=EPSILON*Math.max(1., this.x, v.x)
    && Math.abs(this.y-v.y)<=EPSILON*Math.max(1., this.y, v.y)
  }

  addMutable(v: Vec2): Vec2 {
    this.x += v.x; this.y += v.y;
    return this;
  }

  subMutable(v: Vec2): Vec2 {
    this.x -= v.x; this.y -= v.y;
    return this;
  }

  mulMutable(f: number): Vec2 {
    this.x *= f; this.y *= f;
    return this;
  }

  divMutable(f: number): Vec2 {
    this.x /= f; this.y /= f;
    return this;
  }

  mulElMutable(v: Vec2): Vec2 {
    this.x *= v.x; this.y *= v.y;
    return this;
  }

  divElMutable(v: Vec2): Vec2 {
    this.x /= v.x; this.y /= v.y;
    return this;
  }

  normalizeMutable(): Vec2 { return this.divMutable(Math.sqrt(this.dot(this))); }

  add(v: Vec2): Vec2 { return this.copy().addMutable(v); }

  sub(v: Vec2): Vec2 { return this.copy().subMutable(v); }

  mul(f: number): Vec2 { return this.copy().mulMutable(f); }

  div(f: number): Vec2 { return this.copy().divMutable(f); }

  mulEl(v: Vec2): Vec2 { return this.copy().mulElMutable(v); }

  divEl(v: Vec2): Vec2 { return this.copy().divElMutable(v); }

  normalize(): Vec2 { return this.copy().normalizeMutable(); }

  length(): number { return Math.sqrt(this.dot(this)); }

  dot(v: Vec2): number { return this.x*v.x + this.y*v.y; }

  copy(): Vec2 { return new Vec2(this.x, this.y); }

  floor(): Vec2 { return new Vec2(Math.floor(this.x), Math.floor(this.y)); }

  fract(): Vec2 { return this.copy().subMutable(this.floor()); }

  getArray(): number[] { return [this.x, this.y]; }

  /** Векторное произведение */
  cross(v: Vec2): number { return this.x*v.y - this.y*v.x; }
}

/****************************************************************************** 
 * Класс двумерной матрицы 
 * */
export class Mat2 implements IMatrixes<Mat2, Vec2> {
  i: Vec2; j: Vec2;
  static ID = () => new Mat2(Vec2.I(), Vec2.J());
  static ZERO = () => new Mat2(Vec2.ZERO(), Vec2.ZERO());

  cx(): Vec2 { return new Vec2(this.i.x, this.j.x); }
  cy(): Vec2 { return new Vec2(this.i.y, this.j.y); }

  constructor(i: Vec2, j: Vec2) {
    this.i = i.copy(); this.j = j.copy();
  }

  equals(v: Mat2): boolean {
    return this.i.equals(v.i) && this.j.equals(v.j);
  }

  copy(): Mat2 {
    return new Mat2(this.i.copy(), this.j.copy());  
  }

  transpose(): Mat2 { return new Mat2(this.cx(), this.cy()); }

  addMatMutable(m: Mat2): Mat2 {
    this.i.addMutable(m.i);
    this.j.addMutable(m.j);
    return this;
  }

  subMatMutable(m: Mat2): Mat2 {
    this.i.subMutable(m.i);
    this.j.subMutable(m.j);
    return this;
  }

  mulMutable(n: number): Mat2 {
    this.i.mulMutable(n);
    this.j.mulMutable(n);
    return this;
  }

  divMutable(n: number): Mat2 {
    this.i.divMutable(n);
    this.j.divMutable(n);
    return this;
  }

  addMat(m: Mat2): Mat2 { return this.copy().addMatMutable(m); }

  subMat(m: Mat2): Mat2 { return this.copy().subMatMutable(m); }

  mul(n: number): Mat2 { return this.copy().mulMutable(n); }

  div(n: number): Mat2 { return this.copy().divMutable(n); }

  mulMat(m: Mat2): Mat2 {
    return new Mat2(
      new Vec2(this.cx().dot(m.i), this.cy().dot(m.i)),
      new Vec2(this.cx().dot(m.j), this.cy().dot(m.j))
    );
  }

  mulMatLeft(m: Mat2): Mat2 {
    return new Mat2(
      new Vec2(this.i.dot(m.cx()), this.i.dot(m.cy())),
      new Vec2(this.j.dot(m.cx()), this.j.dot(m.cy()))
    );
  }

  mulVec(v: Vec2): Vec2 {
    return new Vec2(this.cx().dot(v), this.cy().dot(v));
  }

  mulVecLeft(v: Vec2): Vec2 { return new Vec2(this.i.dot(v), this.j.dot(v)); }

  getArray(): number[] {
    return [
      this.i.x, this.i.y,
      this.j.x, this.j.y
    ]
  }

  /**
   * Получить матрицу вращения
   * @param rad - угол поворота
   * @returns 
   */
  rotationMat(rad: number): Mat2 {
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    return new Mat2(new Vec2(c, s), new Vec2(-s, c));
  }

}
