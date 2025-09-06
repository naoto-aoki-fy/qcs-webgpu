// --- single ---
struct GateU {
  stride : u32,   // 2^target
  npairs : u32,   // N/2
  _pad0  : u32,
  _pad1  : u32,
  a : {{complexField}},
  b : {{complexField}},
  c : {{complexField}},
  d : {{complexField}},
};

struct Buf { data: array<{{scalar}}>, };

@group(0) @binding(0) var<storage, read>       stateIn  : Buf;
@group(0) @binding(1) var<storage, read_write> stateOut : Buf;
@group(0) @binding(2) var<uniform>             U       : GateU;

fn cmul(x: vec2<{{scalar}}>, y: vec2<{{scalar}}>) -> vec2<{{scalar}}> {
  return vec2<{{scalar}}>(x.x*y.x - x.y*y.y, x.x*y.y + x.y*x.x);
}

@compute @workgroup_size({{WG_SIZE}})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let k = gid.x;
  if (k >= U.npairs) { return; }

  let s = U.stride;
  let block = (k / s) * (2u * s);
  let i0    = block + (k % s);
  let i1    = i0 + s;

  let f0 = i0 * 2u;
  let f1 = i1 * 2u;

  let v0 = vec2<{{scalar}}>(stateIn.data[f0],    stateIn.data[f0+1u]);
  let v1 = vec2<{{scalar}}>(stateIn.data[f1],    stateIn.data[f1+1u]);

  let a = U.a{{access}};
  let b = U.b{{access}};
  let c = U.c{{access}};
  let d = U.d{{access}};

  let o0 = cmul(a, v0) + cmul(b, v1);
  let o1 = cmul(c, v0) + cmul(d, v1);

  stateOut.data[f0]    = o0.x;
  stateOut.data[f0+1u] = o0.y;
  stateOut.data[f1]    = o1.x;
  stateOut.data[f1+1u] = o1.y;
}

// --- cnot ---
struct CNOTU {
  stride      : u32, // 2^target
  npairs      : u32, // N/2
  controlMask : u32, // 2^control
  _pad        : u32,
};

struct BufC { data: array<{{scalar}}>, };

@group(0) @binding(0) var<storage, read>       stateInC  : BufC;
@group(0) @binding(1) var<storage, read_write> stateOutC : BufC;
@group(0) @binding(2) var<uniform>             Uc        : CNOTU;

@compute @workgroup_size({{WG_SIZE}})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let k = gid.x;
  if (k >= Uc.npairs) { return; }

  let s = Uc.stride;
  let block = (k / s) * (2u * s);
  let i0    = block + (k % s);
  let i1    = i0 + s;

  let f0 = i0 * 2u;
  let f1 = i1 * 2u;

  if ((i0 & Uc.controlMask) != 0u) {
    // swap
    stateOutC.data[f0]    = stateInC.data[f1];
    stateOutC.data[f0+1u] = stateInC.data[f1+1u];
    stateOutC.data[f1]    = stateInC.data[f0];
    stateOutC.data[f1+1u] = stateInC.data[f0+1u];
  } else {
    // copy
    stateOutC.data[f0]    = stateInC.data[f0];
    stateOutC.data[f0+1u] = stateInC.data[f0+1u];
    stateOutC.data[f1]    = stateInC.data[f1];
    stateOutC.data[f1+1u] = stateInC.data[f1+1u];
  }
}
