import { Mat3, Mat4, Quaternion, Vec3, Vec4 } from "./vectors";
import { mat3, mat4, quat, vec3 } from "gl-matrix";

type TestParameters = { name: string, arg: any }[]


/**
 * Функция для замыкания аргументов внутри функции без аргументов
 * @param f функция для вызова
 * @param args аргументы, передаваемые в функцию f
 * @returns функция без аргументов
 */
function callf(f: (...args: any[])=>void, ...args:any[]) {
  return ()=>{ f(...args) }
}

function test(f: (...args: any[])=>void, args: TestParameters) {
  // наименование теста
  let title = 'test';
  let newargs: any[] = [];
  // параметры теста
  for(let i=0; i<args.length; i++) {
    title += ` ${args[i].name}=${args[i].arg}`;
    newargs.push(args[i].arg);
  }
  it(title, callf (f, ...newargs)
  )
}

describe('Mat4.fromAxisAngle', ()=>{
  for(let i = 0; i < 100; i++) {
    const axis = Vec3.RAND();
    const theta = Math.random()*100.;
    test((a: Vec3, t:number)=>{
      const re = Mat4.fromAxisAngle(a,t)
      const m = mat4.create();
      const v = vec3.create();
      vec3.set(v,a.x,a.y,a.z);
      mat4.rotate(m,m,t,v);
      const ex = new Mat4(
        new Vec4(m[0],m[1],m[2],m[3]),
        new Vec4(m[4],m[5],m[6],m[7]),
        new Vec4(m[8],m[9],m[10],m[11]),
        new Vec4(m[12],m[13],m[14],m[15])
      );
      const equal = re.equals(ex)
      expect(equal).toEqual(true)
    }, [
        { name: 'axis', arg: axis },
        { name: 'theta', arg: theta }
      ]
    )
  }
})

describe('Vec3.normalize', ()=>{
  for(let i = 0; i < 100; i++) {
    const v = Vec3.RAND();
    test((a: Vec3)=>{
      const re = a.normalize();
      const len = Math.sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
      const ex = new Vec3(a.x/len,a.y/len,a.z/len);
      expect(re).toEqual(ex)
    }, [{name: 'vector', arg: v}]);
  }
})

describe('Mat4.perspectiveGl', ()=>{
  for(let i = 0; i < 100; i++) {
    const fovy = Math.random()*0.5;
    const aspect = Math.random()*2;
    const near = Math.random();
    const far = 1 + Math.random()*10;
    test((f: number, a: number, zn: number, zf: number)=>{
      const re = Mat4.perspectiveGl(f,a,zn,zf);
      const m = mat4.create();
      mat4.perspectiveNO(m,f,a,zn,zf);
      const ex = new Mat4(
        new Vec4(m[0],m[1],m[2],m[3]),
        new Vec4(m[4],m[5],m[6],m[7]),
        new Vec4(m[8],m[9],m[10],m[11]),
        new Vec4(m[12],m[13],m[14],m[15])
      );
      const equal = re.equals(ex)
      expect(equal).toEqual(true)
    }, [
      {name: 'fovy', arg: fovy},
      {name: 'aspect', arg: aspect},
      {name: 'near', arg: near},
      {name: 'far', arg: far}
    ]);
  }
})

describe('Mat4.perspectiveDx', ()=>{
  for(let i = 0; i < 100; i++) {
    const fovy = Math.random()*0.5;
    const aspect = Math.random()*2;
    const near = Math.random();
    const far = 1 + Math.random()*10;
    test((f: number, a: number, zn: number, zf: number)=>{
      const re = Mat4.perspectiveDx(f,a,zn,zf);
      const m = mat4.create();
      mat4.perspectiveZO(m,f,a,zn,zf);
      const ex = new Mat4(
        new Vec4(m[0],m[1],m[2],m[3]),
        new Vec4(m[4],m[5],m[6],m[7]),
        new Vec4(m[8],m[9],m[10],m[11]),
        new Vec4(m[12],m[13],m[14],m[15])
      );
      const equal = re.equals(ex)
      expect(equal).toEqual(true)
    }, [
      {name: 'fovy', arg: fovy},
      {name: 'aspect', arg: aspect},
      {name: 'near', arg: near},
      {name: 'far', arg: far}
    ]);
  }
})

describe('Mat4.lookAt', ()=>{
  for(let i = 0; i < 100; i++) {
    const from = Vec3.RAND().mulMutable(100);
    const to = Vec3.RAND().mulMutable(100);
    const up = Vec3.RAND().normalize();
    test((f: Vec3, t: Vec3, u: Vec3)=>{
      const re = Mat4.lookAt(f,t,u);
      const m = mat4.create();
      const from = vec3.create();
      vec3.set(from, f.x, f.y, f.z);
      const to = vec3.create();
      vec3.set(to, t.x, t.y, t.z);
      const up = vec3.create();
      vec3.set(up, u.x, u.y, u.z);
      mat4.lookAt(m,from,to,up);
      const ex = new Mat4(
        new Vec4(m[0],m[1],m[2],m[3]),
        new Vec4(m[4],m[5],m[6],m[7]),
        new Vec4(m[8],m[9],m[10],m[11]),
        new Vec4(m[12],m[13],m[14],m[15])
      );
      const equal = re.equals(ex);
      expect(equal).toBeTrue();
      //expect(re).toEqual(ex)
    }, [
      {name: 'from', arg: from},
      {name: 'to', arg: to},
      {name: 'up', arg: up}
    ]);
  }
})


describe('Mat4.mulMat', ()=>{
  for(let i = 0; i < 100; i++) {
    const m1 = Mat4.RAND().mulMutable(100);
    const m2 = Mat4.RAND().mulMutable(100);
    test((m1: Mat4, m2: Mat4)=>{
      const re = m1.mulMat(m2);
      const mt1 = mat4.create();
      mat4.set(mt1,
        m1.i.x, m1.i.y, m1.i.z, m1.i.w,
        m1.j.x, m1.j.y, m1.j.z, m1.j.w,
        m1.k.x, m1.k.y, m1.k.z, m1.k.w,
        m1.l.x, m1.l.y, m1.l.z, m1.l.w
        );
      const mt2 = mat4.create();
      mat4.set(mt2,
        m2.i.x, m2.i.y, m2.i.z, m2.i.w,
        m2.j.x, m2.j.y, m2.j.z, m2.j.w,
        m2.k.x, m2.k.y, m2.k.z, m2.k.w,
        m2.l.x, m2.l.y, m2.l.z, m2.l.w
        );
      const m = mat4.create();
      mat4.multiply(m, mt1, mt2);
      const ex = new Mat4(
        new Vec4(m[0],m[1],m[2],m[3]),
        new Vec4(m[4],m[5],m[6],m[7]),
        new Vec4(m[8],m[9],m[10],m[11]),
        new Vec4(m[12],m[13],m[14],m[15])
      );
      const equal = re.equals(ex);
      expect(equal).toBeTrue();
      //expect(re).toEqual(ex)
    }, [
      {name: 'Mat4', arg: m1},
      {name: 'Mat4', arg: m2}
    ]);
  }
})


describe('Mat4.transpose', ()=>{
  for(let i = 0; i < 100; i++) {
    const m1 = Mat4.RAND().mulMutable(100);
    test((m1: Mat4)=>{
      const re = m1.transpose();
      const mt = mat4.create();
      mat4.set(mt,
        m1.i.x, m1.i.y, m1.i.z, m1.i.w,
        m1.j.x, m1.j.y, m1.j.z, m1.j.w,
        m1.k.x, m1.k.y, m1.k.z, m1.k.w,
        m1.l.x, m1.l.y, m1.l.z, m1.l.w
        );
      const m = mat4.create();
      mat4.transpose(m, mt);
      const ex = new Mat4(
        new Vec4(m[0],m[1],m[2],m[3]),
        new Vec4(m[4],m[5],m[6],m[7]),
        new Vec4(m[8],m[9],m[10],m[11]),
        new Vec4(m[12],m[13],m[14],m[15])
      );
      const equal = re.equals(ex);
      expect(equal).toBeTrue();
      //expect(re).toEqual(ex)
    }, [
      {name: 'Mat4', arg: m1}
    ]);
  }
})

describe('Mat4.getQuaternion', ()=>{
  for(let i = 0; i < 100; i++) {
    const q1 = Quaternion.random();
    const m1 = Mat4.fromQuat(q1);
    test((m1: Mat4)=>{
      const re = m1.getQuaternion();
      const mt = mat4.create();
      mat4.set(mt,
        m1.i.x, m1.i.y, m1.i.z, m1.i.w,
        m1.j.x, m1.j.y, m1.j.z, m1.j.w,
        m1.k.x, m1.k.y, m1.k.z, m1.k.w,
        m1.l.x, m1.l.y, m1.l.z, m1.l.w
        );
      const q = quat.create();
      mat4.getRotation(q, mt);
      const ex = new Quaternion(q[0],q[1],q[2],q[3]);
      const equal = re.equals(ex);
      expect(equal).toBeTrue();
      //expect(re).toEqual(ex)
    }, [
      {name: 'Mat4', arg: m1}
    ]);
  }
})


describe('Quaternion.fromLookAt', ()=>{
  for(let i = 0; i < 100; i++) {
    const from = Vec3.RAND().mulMutable(100);
    const to = Vec3.RAND().mulMutable(100);
    const up = Vec3.RAND().normalize();
    test((f: Vec3, t: Vec3, u: Vec3)=>{
      const re = Quaternion.fromLookAt(f,t,u);
      const m = mat4.create();
      const from = vec3.create();
      vec3.set(from, f.x, f.y, f.z);
      const to = vec3.create();
      vec3.set(to, t.x, t.y, t.z);
      const up = vec3.create();
      vec3.set(up, u.x, u.y, u.z);
      mat4.lookAt(m,from,to,up);
      const q = quat.create();
      mat4.getRotation(q, m);
      const ex = new Quaternion(q[0],q[1],q[2],q[3]);
      const equal = re.equals(ex);
      expect(equal).toBeTrue();
      //expect(re).toEqual(ex)
    }, [
      {name: 'from', arg: from},
      {name: 'to', arg: to},
      {name: 'up', arg: up}
    ]);
  }
})

