'use strict';
const { readContainer, loadTensors, ByteLevelBPE, QOmniJS, loadWasmModel, initGPU, QOmniGPU } = window.QOmniLib;
const $ = (id) => document.getElementById(id);
let loaded = null, jsModel = null, wasmModel = null, gpuModel = null, tok = null, backend = 'none', generating = false, stopFlag = false;

function log(msg) { $('status').textContent = msg; }

// fetch with a hard timeout (so a genuinely stuck request surfaces as a visible
// error instead of an indefinite "loading…") and streamed progress reporting.
async function fetchWithProgress(url, onProgress, timeoutMs = 45000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new Error(`timed out after ${timeoutMs / 1000}s`)), timeoutMs);
  let resp;
  try { resp = await fetch(url, { signal: ctrl.signal }); }
  finally { clearTimeout(timer); }
  if (!resp.ok) throw new Error(`fetch ${url} failed: ${resp.status}`);
  const total = Number(resp.headers.get('content-length')) || 0;
  if (!resp.body) return new Uint8Array(await resp.arrayBuffer()).buffer;
  const reader = resp.body.getReader();
  const chunks = []; let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); received += value.byteLength;
    if (onProgress) onProgress(received, total);
  }
  const out = new Uint8Array(received); let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.byteLength; }
  return out.buffer;
}

async function loadModel() {
  const t0 = performance.now();
  const progress = (label) => (got, total) => {
    const mb = (got / 1e6).toFixed(1);
    log(total ? `${label} ${mb}/${(total / 1e6).toFixed(1)} MB (${(100 * got / total).toFixed(0)}%)…` : `${label} ${mb} MB…`);
  };
  log('downloading model.qpack (33 MB)…');
  const [buf, wasmBytesRaw] = await Promise.all([
    fetchWithProgress('./model.qpack', progress('downloading model.qpack')),
    fetchWithProgress('./kernel.wasm', progress('downloading kernel.wasm')).catch((e) => { console.warn('kernel.wasm fetch failed:', e); return null; }),
  ]);
  const wasmBytes = wasmBytesRaw;
  log(`parsing container (${(buf.byteLength / 1e6).toFixed(1)} MB downloaded in ${((performance.now() - t0) / 1000).toFixed(1)}s)…`);
  const c = readContainer(buf);
  loaded = loadTensors(c);
  tok = new ByteLevelBPE(c.json('tokenizer.json'));
  jsModel = new QOmniJS(loaded);
  $('params').textContent = `${loaded.cfg.num_hidden_layers} layers · hidden ${loaded.cfg.hidden_size} · vocab ${loaded.cfg.vocab_size.toLocaleString()}`;

  log("running self-test (compares to the reference model's saved logits)…");
  const st = loaded.selftest;
  let ok = 0;
  if (st) {
    for (const cs of st.cases) {
      jsModel.reset(); let lg;
      for (const id of cs.ids) lg = jsModel.step(id);
      const order = [...lg.keys()].sort((a, b) => lg[b] - lg[a]).slice(0, 5);
      if (order[0] === cs.top5[0] && Math.abs(lg[cs.top5[0]] - cs.top5_logits[0]) < 5e-2) ok++;
    }
    $('selftest').textContent = `self-test: ${ok}/${st.cases.length} ${ok === st.cases.length ? '✓' : '✗ (results may be unreliable)'}`;
  }
  jsModel.reset();

  // Preferred backend: WebAssembly (SIMD), a compiled C kernel — synchronous, and checked token-for-token identical
  // to the JS reference before shipping (see kernel.c's header comment). Falls back to WebGPU, then plain JS.
  if (wasmBytes) {
    try { log('compiling the WebAssembly kernel and uploading weights…'); wasmModel = await loadWasmModel(loaded, wasmBytes); backend = 'wasm'; }
    catch (e) { console.warn('WASM init failed:', e); }
  }
  if (backend === 'none' && navigator.gpu) {
    try { log('initialising WebGPU (experimental — uploading weights to the GPU)…'); const gpuCtx = await initGPU(); gpuModel = new QOmniGPU(gpuCtx, loaded); backend = 'webgpu'; }
    catch (e) { console.warn('WebGPU init failed, using CPU:', e); }
  }
  if (backend === 'none') backend = 'js';
  const labels = { wasm: 'WebAssembly + SIMD (compiled C kernel, ~30 tok/s single-threaded)', webgpu: 'WebGPU (experimental, GPU-accelerated)', js: 'plain JavaScript fallback (slow, ~1-2 tok/s)' };
  $('backend').textContent = labels[backend];
  log('ready.');
  $('gen').disabled = false;
  $('gpuToggle').disabled = !(navigator.gpu && backend !== 'webgpu');
}

async function useGpuInstead() {
  if (gpuModel || !navigator.gpu) return;
  log('initialising WebGPU (experimental)…');
  try { const gpuCtx = await initGPU(); gpuModel = new QOmniGPU(gpuCtx, loaded); backend = 'webgpu'; $('backend').textContent = 'WebGPU (experimental, GPU-accelerated)'; log('ready.'); }
  catch (e) { log('WebGPU init failed: ' + e.message); }
}

function suppressNonText(logits, vocabTextEnd) {
  for (let i = vocabTextEnd; i < logits.length; i++) logits[i] = -Infinity;
  return logits;
}

function sample(logits, temperature, topK) {
  const idx = [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, topK);
  if (temperature <= 0) return idx[0];
  const vals = idx.map((i) => logits[i] / temperature);
  const mx = Math.max(...vals); const exps = vals.map((v) => Math.exp(v - mx)); const sum = exps.reduce((a, b) => a + b, 0);
  let r = Math.random() * sum, acc = 0;
  for (let k = 0; k < idx.length; k++) { acc += exps[k]; if (r <= acc) return idx[k]; }
  return idx[0];
}

async function generate() {
  if (generating) { stopFlag = true; return; }
  generating = true; stopFlag = false; $('gen').textContent = 'Stop';
  const prompt = $('prompt').value || 'Once upon a time,';
  const maxNew = parseInt($('maxnew').value, 10) || 60;
  const temp = parseFloat($('temp').value);
  const model = backend === 'wasm' ? wasmModel : backend === 'webgpu' ? gpuModel : jsModel;
  model.reset();
  const ids = tok.encodeWithBos(prompt);
  $('out').textContent = prompt;
  let logits;
  const t0 = performance.now(); let n = 0;
  for (const id of ids) logits = await model.step(id);
  const textEnd = loaded.cfg.special_offset; // keep sampling in the text vocabulary only
  while (n < maxNew && !stopFlag) {
    const next = sample(suppressNonText(logits.slice(), textEnd), temp, 40);
    if (next === 2 /* <eos> */) break;
    $('out').textContent += tok.decode([next]);
    n++;
    $('tps').textContent = `${(n / ((performance.now() - t0) / 1000)).toFixed(1)} tok/s (${backend})`;
    logits = await model.step(next);
    await new Promise((r) => setTimeout(r, 0)); // yield to the UI thread
  }
  generating = false; $('gen').textContent = 'Generate';
}

$('gen').addEventListener('click', generate);
$('gpuToggle').addEventListener('click', useGpuInstead);
loadModel().catch((e) => { log('error: ' + e.message); console.error(e); });
