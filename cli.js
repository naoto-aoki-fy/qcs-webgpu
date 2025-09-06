#!/usr/bin/env node
import gpu from '@kmamal/gpu'
import { getWGSL, buildSingleUBOData, buildCNOTUBOData, createUBOWithData } from './docs/ghz-lib.js'

async function main() {
  const args = process.argv.slice(2)
  let numQubits = 3
  let precision = 'fp32'
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]
    if ((arg === '-n' || arg === '--qubits') && i + 1 < args.length) {
      numQubits = parseInt(args[++i], 10)
    } else if ((arg === '-p' || arg === '--precision') && i + 1 < args.length) {
      precision = args[++i]
    }
  }
  if (!Number.isInteger(numQubits) || numQubits < 1) {
    console.error('Invalid number of qubits')
    process.exit(1)
  }
  const is64 = precision === 'fp64'
  const floatArray = is64 ? Float64Array : Float32Array
  const bytesPerComplex = is64 ? 16 : 8
  const WG_SIZE = 256

  const instance = gpu.create([])
  const adapter = await instance.requestAdapter()
  const device = await adapter.requestDevice()

  const Nstates = 1 << numQubits
  const stateBytes = Nstates * bytesPerComplex

  const stateA = device.createBuffer({
    size: stateBytes,
    usage: gpu.GPUBufferUsage.STORAGE | gpu.GPUBufferUsage.COPY_SRC | gpu.GPUBufferUsage.COPY_DST,
  })
  const stateB = device.createBuffer({
    size: stateBytes,
    usage: gpu.GPUBufferUsage.STORAGE | gpu.GPUBufferUsage.COPY_SRC | gpu.GPUBufferUsage.COPY_DST,
  })
  const readback = device.createBuffer({
    size: stateBytes,
    usage: gpu.GPUBufferUsage.COPY_DST | gpu.GPUBufferUsage.MAP_READ,
  })

  const init = new floatArray(Nstates * 2)
  init[0] = 1
  device.queue.writeBuffer(stateA, 0, init)

  const { single: singleQubitWGSL, cnot: cnotWGSL } = await getWGSL(is64, WG_SIZE)

  const singlePipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: singleQubitWGSL }), entryPoint: 'main' }
  })
  const cnotPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: cnotWGSL }), entryPoint: 'main' }
  })

  async function runSingle(inBuf, outBuf, target, mat) {
    const { buf } = buildSingleUBOData(target, Nstates, mat, is64)
    const ubo = createUBOWithData(device, buf, gpu)
    const bindGroup = device.createBindGroup({
      layout: singlePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inBuf } },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: ubo } },
      ],
    })
    const numPairs = Nstates >>> 1
    const numWorkgroups = Math.ceil(numPairs / WG_SIZE)
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(singlePipe)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(numWorkgroups)
    pass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
  }

  async function runCNOT(inBuf, outBuf, control, target) {
    const { buf } = buildCNOTUBOData(control, target, Nstates)
    const ubo = createUBOWithData(device, buf, gpu)
    const bindGroup = device.createBindGroup({
      layout: cnotPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inBuf } },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: ubo } },
      ],
    })
    const numPairs = Nstates >>> 1
    const numWorkgroups = Math.ceil(numPairs / WG_SIZE)
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(cnotPipe)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(numWorkgroups)
    pass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
  }

  const INV_SQRT2 = 1 / Math.sqrt(2)
  const H = {
    a: [INV_SQRT2, 0],
    b: [INV_SQRT2, 0],
    c: [INV_SQRT2, 0],
    d: [-INV_SQRT2, 0],
  }

  let bufIn = stateA, bufOut = stateB
  await runSingle(bufIn, bufOut, 0, H)
  ;[bufIn, bufOut] = [bufOut, bufIn]
  for (let t = 1; t < numQubits; t++) {
    await runCNOT(bufIn, bufOut, 0, t)
    ;[bufIn, bufOut] = [bufOut, bufIn]
  }

  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(bufIn, 0, readback, 0, stateBytes)
  device.queue.submit([enc.finish()])
  await device.queue.onSubmittedWorkDone()
  await readback.mapAsync(gpu.GPUMapMode.READ)
  const arr = new floatArray(readback.getMappedRange()).slice()
  readback.unmap()

  let norm = 0
  for (let i = 0; i < arr.length; i += 2) {
    const re = arr[i], im = arr[i+1]
    norm += re*re + im*im
  }
  const amp0 = [arr[0], arr[1]]
  const lastIdx = (Nstates - 1) * 2
  const amp1 = [arr[lastIdx], arr[lastIdx+1]]
  console.log(`|0...0> amp: ${amp0[0]} ${amp0[1]}i`)
  console.log(`|1...1> amp: ${amp1[0]} ${amp1[1]}i`)
  console.log(`norm: ${norm}`)

  device.destroy()
  gpu.destroy(instance)
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
