const UBO_SIZE = 256; // 256-byte aligned UBOs

let cachedWGSL = null;

async function loadWGSLFile() {
  if (cachedWGSL !== null) { return cachedWGSL; }
  if (typeof window === 'undefined') {
    const fs = await import('node:fs/promises');
    cachedWGSL = await fs.readFile(new URL('./ghz.wgsl', import.meta.url), 'utf8');
  } else {
    const res = await fetch(new URL('./ghz.wgsl', import.meta.url));
    cachedWGSL = await res.text();
  }
  return cachedWGSL;
}

export async function getWGSL(is64, WG_SIZE) {
  const text = await loadWGSLFile();
  const parts = text.split(/\/\/ --- cnot ---/);
  const singlePart = parts[0].replace(/\/\/ --- single ---/, '');
  const cnotPart = parts[1];
  const scalar = is64 ? 'f64' : 'f32';
  const complexField = is64 ? `vec2<${scalar}>` : `vec4<${scalar}>`;
  const access = is64 ? '' : '.xy';
  const replacers = [
    ['{{scalar}}', scalar],
    ['{{complexField}}', complexField],
    ['{{access}}', access],
    ['{{WG_SIZE}}', WG_SIZE.toString()],
  ];
  let single = singlePart;
  let cnot = cnotPart;
  for (const [k, v] of replacers) {
    single = single.replaceAll(k, v);
    cnot = cnot.replaceAll(k, v);
  }
  return { single, cnot };
}

export function buildSingleUBOData(target, N, mat, is64) {
  const buf = new ArrayBuffer(UBO_SIZE);
  const dv = new DataView(buf);
  const stride = 1 << target;
  const npairs = N >>> 1;
  dv.setUint32(0, stride, true);
  dv.setUint32(4, npairs, true);
  dv.setUint32(8, 0, true);
  dv.setUint32(12, 0, true);
  if (is64) {
    const f64 = new Float64Array(buf);
    let base = 2;
    f64[base+0]=mat.a[0]; f64[base+1]=mat.a[1];
    f64[base+2]=mat.b[0]; f64[base+3]=mat.b[1];
    f64[base+4]=mat.c[0]; f64[base+5]=mat.c[1];
    f64[base+6]=mat.d[0]; f64[base+7]=mat.d[1];
  } else {
    const f32 = new Float32Array(buf);
    let base = 4; // float index
    f32[base+0]=mat.a[0]; f32[base+1]=mat.a[1];
    f32[base+4]=mat.b[0]; f32[base+5]=mat.b[1];
    f32[base+8]=mat.c[0]; f32[base+9]=mat.c[1];
    f32[base+12]=mat.d[0]; f32[base+13]=mat.d[1];
  }
  return { buf, stride, npairs };
}

export function buildCNOTUBOData(control, target, N) {
  const buf = new ArrayBuffer(UBO_SIZE);
  const dv = new DataView(buf);
  const stride = 1 << target;
  const npairs = N >>> 1;
  const controlMask = 1 << control;
  dv.setUint32(0, stride, true);
  dv.setUint32(4, npairs, true);
  dv.setUint32(8, controlMask, true);
  dv.setUint32(12, 0, true);
  return { buf, stride, npairs, controlMask };
}

export function createUBOWithData(device, arrayBuffer, gpuAPI) {
  const usage = gpuAPI ? gpuAPI.GPUBufferUsage : GPUBufferUsage;
  const ubo = device.createBuffer({
    size: UBO_SIZE,
    usage: usage.UNIFORM | usage.COPY_DST,
  });
  device.queue.writeBuffer(ubo, 0, arrayBuffer);
  return ubo;
}

export { UBO_SIZE };
