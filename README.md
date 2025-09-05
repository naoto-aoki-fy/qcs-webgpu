# WebGPU Quantum Simulator (State-Vector)

A tiny, dependency-free, **in-browser quantum state-vector simulator** powered by **WebGPU** compute shaders and **WGSL**. It builds a GHZ state with a simple circuit (`H` on qubit 0, followed by a CNOT fan-out) and lets you vary the number of qubits and numeric precision (**FP32/FP64**). The UI is bilingual (**English / 日本語**) and everything runs locally in your browser—no server, no bundler.

## Features

* **WebGPU compute** state-vector simulation (no WebGL, no WASM)
* **Circuit demo:** GHZ preparation `H(0)` → `CNOT(0→1)` … `CNOT(0→n-1)`
* **Gates included:** arbitrary **single-qubit 2×2 complex** gate, **CNOT**
* **Precision toggle:** **FP32** (fast, default) or **FP64** (requires `shader-f64`)
* **Device/limit checks:** refuses runs that exceed `maxStorageBufferBindingSize`
* **Bilingual UI:** English / 日本語 with auto-detection and manual switch
* **Readable logs:** rich DevTools output (device info, timings, UBO dumps, previews)
* **No build step:** a single `index.html`

## Quick Start

### Requirements

* A browser with **WebGPU**: **Chrome/Edge 125+** (desktop).
* **Secure context**: use `https://`.

### Use

1. Choose **Language**, **Number of qubits**, and **Precision (FP32/FP64)**.
2. Click **Run**.
3. The **Summary** shows amplitudes of `|0…0⟩` and `|1…1⟩`, the **norm** `∥ψ∥²`, and **runtime**.
4. The **Output** prints either the full state vector (for small `n`) or non-zeros above a threshold (for large `n`).
5. Open **DevTools** to see detailed logs (device limits, UBO headers, pair mapping samples, timers).

> **Expected result (GHZ):** only `|0…0⟩` and `|1…1⟩` have non-zero amplitudes, each ≈ `1/√2`.

## How it Works

### Data model

* **State vector** of size `N = 2^n` stored as interleaved complex scalars:

  * FP32: `Float32Array [re0, im0, re1, im1, …]` (8 bytes per amplitude)
  * FP64: `Float64Array [re0, im0, re1, im1, …]` (16 bytes per amplitude)
* **Ping-pong buffers**: `stateA` (in), `stateB` (out), swapped after each kernel.

### Compute pipelines (WGSL)

#### 1) Single-qubit gate (`2×2` complex)

For a **target** qubit `t`, pairs of basis indices that differ only in bit `t` are processed:

* `stride = 2^t`
* For each pair `(i0, i1)`, apply:

  ```
  |ψ'⟩[i0] = a·ψ[i0] + b·ψ[i1]
  |ψ'⟩[i1] = c·ψ[i0] + d·ψ[i1]
  ```
* Uniforms (`GateU`): `stride`, `npairs = N/2`, and matrix entries `a, b, c, d`.

**Alignment trick:**
Uniform buffers need 16-byte alignment. The implementation uses:

* **FP64:** `vec2<f64>` for each complex number.
* **FP32:** `vec4<f32>` and accesses `.xy` (re, im), leaving `.zw` as padding.

#### 2) CNOT (`control → target`)

* Same pair mapping by `target`, with a bitmask for the `control`.
* If `(i0 & controlMask) != 0`, swap the two amplitudes; otherwise copy.

### Work distribution

* **Workgroup size:** `WG_SIZE = min(256, device.limits.maxComputeInvocationsPerWorkgroup)`
* **Dispatch:** `numPairs = N/2`, `numWorkgroups = ceil(numPairs / WG_SIZE)`

### Circuit (demo)

1. Apply **Hadamard** `H` on qubit 0.
2. For each `t = 1 … n−1`, apply **CNOT** `0 → t`.
   This produces the GHZ state `( |0…0⟩ + |1…1⟩ ) / √2`.

## UI & Localization

* Language auto-detects (`navigator.language`) and can be switched manually.
* Controls:

  * **Number of qubits** (`n`, min `1`)
  * **Precision:** **FP32** or **FP64**
  * **Run** button (auto-run is intentionally disabled)
* Summary displays:

  * `n`, `2^n` (state dimension), amplitudes of `⟨0…0|` and `⟨1…1|`, `∥ψ∥²`, and total simulation time.

## Limits & Performance

* **Memory growth:** state size is `O(2^n)`.

  * Bytes per amplitude = **8 (FP32)** or **16 (FP64)**.
  * The simulator enforces `device.limits.maxStorageBufferBindingSize` and computes a safe **max qubit count**:

    ```
    max_n ≈ floor( log2( maxStorageBufferBindingSize / bytesPerAmplitude ) )
    ```
* **FP64 availability:** requires `shader-f64`. Not all devices expose it; the app gracefully warns when unsupported.
* **Tips for speed:**

  * Prefer **FP32** unless you specifically need FP64.
  * Reduce `n`.
  * Close other GPU-heavy tabs/apps.
  * Keep DevTools open to see timings (`performance.now()`, console `time`).
