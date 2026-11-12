// Web Worker hosting a Llama-family model (Qwen2 / Llama / SmolLM...) from a QPACK2 container (f32 / q8 / q4 tensors)
// on the llama_core.wasm decode step. Same chat protocol as nq_worker.js, so one page can switch between the two:
// Messages in:  {type:'load', url, version, size?}  {type:'chat', turns:[{role,content}], text, opts}  {type:'regenerate', opts}
//               {type:'stop'}  {type:'reset'}
// Messages out: {type:'progress', got, total | cached}, {type:'status'}, {type:'ready'}, {type:'seg', i, kind:'text', text},
//               {type:'done', tokens, ms, hitLimit, raw, ctx}, {type:'error', message}
// The KV cache is reused across turns: a new prompt only runs the tokens after its common prefix with the cached ones.
'use strict';
self.window = self;
const V = new URL(self.location.href).searchParams.get('v') || '1';
importScripts(`qpack.js?v=${V}`, `hf_tokenizer.js?v=${V}`);
const Lib = self.QOmniLib, HF = self.QPack;
const MAX_CTX = 2048;
let core = null, mem = null, cfg = null, tok = null, stop = false;
let cached = [];                // token ids whose K/V are in the cache (positions 0..cached.length-1)
let lastPrompt = null;          // prompt ids of the last answer (regenerate)
let stopIds = new Set(), banned = [];

const yieldChannel = new MessageChannel(); const yieldQueue = [];
yieldChannel.port1.onmessage = () => { const r = yieldQueue.shift(); if (r) r(); };
const yieldNow = () => new Promise((r) => { yieldQueue.push(r); yieldChannel.port2.postMessage(0); });

async function fetchCached(url, version, size) {
  const key = `${url}#${version}`;
  try {
    const cache = await caches.open('nq-models');
    const hit = await cache.match(key);
    if (hit) { postMessage({ type: 'progress', cached: true }); return await hit.arrayBuffer(); }
    const resp = await fetch(url); if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
    const total = Number(resp.headers.get('content-length')) || size || 0, reader = resp.body.getReader(), chunks = []; let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.byteLength; postMessage({ type: 'progress', got, total }); }
    const buf = new Uint8Array(got); let off = 0; for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
    try { for (const k of await cache.keys()) if (k.url.split('#')[0].endsWith(url.split('?')[0])) await cache.delete(k); await cache.put(key, new Response(buf)); } catch (e) {}
    return buf.buffer;
  } catch (e) {
    if (String(e.message).startsWith('fetch')) throw e;
    const resp = await fetch(url); return await resp.arrayBuffer();
  }
}

function f16ToF32(bytes) {
  const n = bytes.length >> 1, out = new Float32Array(n), u = new Uint16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
  for (let i = 0; i < n; i++) {
    const h = u[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    out[i] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

const SLOT = { embed: 0, lm_head: 1, final_norm: 2 };
const LSLOT = { attn_norm: 0, ffn_norm: 1, wq: 2, wk: 3, wv: 4, wo: 5, bq: 6, bk: 7, bv: 8, w_gate: 9, w_up: 10, w_down: 11 };

async function load(url, version, size) {
  const buf = await fetchCached(url, version, size);
  postMessage({ type: 'status', text: 'unpacking' });
  const c = Lib.readContainer(buf), man = c.json('manifest.json');
  cfg = c.json('config.json');
  if (cfg.arch !== 'llama') throw new Error(`not a Llama-family container (arch ${cfg.arch})`);
  tok = new HF.HFTokenizer(c.json('tokenizer.json'));
  const wasm = await (await fetch(`llama_core.wasm?v=${V}`)).arrayBuffer();
  core = (await WebAssembly.instantiate(wasm, {})).instance.exports; mem = core.memory;
  const D = cfg.hidden_size, nh = cfg.num_attention_heads, hd = cfg.head_dim || D / nh;
  const ci = new Int32Array(mem.buffer, core.ll_cfg(), 12), cf = new Float32Array(mem.buffer, core.ll_cfg(), 12);
  ci.set([D, cfg.num_hidden_layers, nh, cfg.num_key_value_heads, hd, cfg.intermediate_size, cfg.vocab_size, MAX_CTX, cfg.tie_embeddings ? 1 : 0, cfg.qkv_bias ? 1 : 0]);
  cf[10] = cfg.rms_norm_eps; cf[11] = cfg.rope_theta;
  const inv = new Float32Array(mem.buffer, core.ll_inv(), 256);
  for (let i = 0; i < hd / 2; i++) inv[i] = 1 / Math.pow(cfg.rope_theta, (2 * i) / hd);
  for (const e of man.tensors) {
    let slot;
    if (e.name in SLOT) slot = SLOT[e.name];
    else { const m = e.name.match(/^layers\.(\d+)\.(\w+)$/); if (!m || !(m[2] in LSLOT)) continue; slot = 8 + Number(m[1]) * 16 + LSLOT[m[2]]; }
    const rows = e.shape[0], cols = e.shape.length > 1 ? e.shape[1] : 1, b = c.blob(e.blob);
    const put = (bytes) => { const p = core.alloc(bytes.length); if (!p) throw new Error('out of WebAssembly memory'); new Uint8Array(mem.buffer, p, bytes.length).set(bytes); return p; };
    const putF = (f) => put(new Uint8Array(f.buffer, f.byteOffset, f.byteLength));
    if (e.kind === 'q4') core.ll_set4(slot, put(b), putF(f16ToF32(c.blob(e.scale))), rows, cols);
    else if (e.kind === 'q8') { const sb = c.blob(e.scale); core.ll_set(slot, put(b), putF(new Float32Array(sb.buffer.slice(sb.byteOffset, sb.byteOffset + sb.length))), rows, cols); }
    else if (e.kind === 'f32') core.ll_set(slot, put(b), 0, rows, cols);
    else if (e.kind === 'f16') core.ll_set(slot, putF(f16ToF32(b)), 0, rows, cols);
    else throw new Error(`tensor kind ${e.kind} is not supported in the browser`);
  }
  if (!core.ll_init()) throw new Error('out of WebAssembly memory (KV cache)');
  const sp = cfg.special_tokens || {};
  stopIds = new Set([cfg.eos_token_id, sp['<|im_end|>'], sp['<|endoftext|>'], sp['<|eot_id|>']].filter((x) => x != null));
  banned = Object.values(sp).filter((id) => !stopIds.has(id));
  let ok = 0, n = 0;
  const st = c.toc['selftest.json'] ? c.json('selftest.json') : null;
  if (st) for (const cs of st.cases) {
    n++; core.ll_reset(); let lg = null; cs.ids.forEach((id, i) => { lg = step(id, i === cs.ids.length - 1); });
    let a = 0; for (let v = 1; v < lg.length; v++) if (lg[v] > lg[a]) a = v;
    ok += a === cs.top5[0] && Math.abs(lg[a] - cs.top5_logits[0]) < 0.25;
  }
  core.ll_reset(); cached = []; lastPrompt = null;
  postMessage({ type: 'ready', selftest: [ok, n], decision: false, layers: cfg.num_hidden_layers, hidden: D, vocab: cfg.vocab_size,
                backend: 'wasm-core', format: `QPACK2 ${man.quant || ''}`.trim(), arch: cfg.model_type || 'llama' });
}

function step(id, want) {
  const p = core.ll_step(id, want ? 1 : 0);
  if (!want) return null;
  if (!p) throw new Error(`context is full (${MAX_CTX} tokens) — start a new chat`);
  return new Float32Array(mem.buffer, p, cfg.vocab_size);
}

const SYSTEM = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.';
function promptIds(turns, text) {
  let s = `<|im_start|>system\n${SYSTEM}<|im_end|>\n`;
  for (const t of turns) s += `<|im_start|>${t.role}\n${t.content}<|im_end|>\n`;
  s += `<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`;
  return tok.encode(s);
}

// run `ids` into the cache, reusing the common prefix with what is already there; returns the logits of the last token
function prefill(ids) {
  let k = 0; while (k < cached.length && k < ids.length - 1 && cached[k] === ids[k]) k++;
  core.ll_set_pos(k); cached = cached.slice(0, k);
  let lg = null;
  for (let i = k; i < ids.length; i++) { lg = step(ids[i], i === ids.length - 1); cached.push(ids[i]); }
  return lg;
}

const scratch = { l: null, idx: new Int32Array(64), val: new Float64Array(64) };
function pick(logits, o, recent) {
  const Vn = logits.length; if (!scratch.l || scratch.l.length !== Vn) scratch.l = new Float32Array(Vn);
  const l = scratch.l; l.set(logits);
  for (const id of banned) if (id < Vn) l[id] = -Infinity;
  for (const t of new Set(recent)) if (!stopIds.has(t)) l[t] = l[t] > 0 ? l[t] / o.rep : l[t] * o.rep;
  if (o.greedy) { let b = 0; for (let i = 1; i < Vn; i++) if (l[i] > l[b]) b = i; return b; }
  const K = 64, idx = scratch.idx, val = scratch.val; let m = 0;
  for (let i = 0; i < Vn; i++) {
    const v = l[i]; if (m === K && v <= val[K - 1]) continue;
    let p = m < K ? m++ : K - 1; while (p > 0 && val[p - 1] < v) { val[p] = val[p - 1]; idx[p] = idx[p - 1]; p--; } val[p] = v; idx[p] = i;
  }
  let sum = 0; const pr = new Float64Array(m); for (let k = 0; k < m; k++) { pr[k] = Math.exp((val[k] - val[0]) / o.temperature); sum += pr[k]; }
  let cum = 0, cut = m; for (let k = 0; k < m; k++) { cum += pr[k] / sum; if (cum >= o.top_p) { cut = k + 1; break; } }
  let tot = 0; for (let k = 0; k < cut; k++) tot += pr[k];
  let r = Math.random() * tot; for (let k = 0; k < cut; k++) { r -= pr[k]; if (r <= 0) return idx[k]; }
  return idx[0];
}

async function answer(prompt, o) {
  stop = false; lastPrompt = prompt;
  let logits = prefill(prompt);
  const t0 = performance.now();   // tok/s of generation (the prompt prefill is not counted)
  const out = []; let n = 0, ended = false;
  while (n < o.max_new && !stop) {
    const next = pick(logits, o, out.slice(-64));
    n++;
    if (stopIds.has(next)) { ended = true; break; }
    out.push(next); cached.push(next);
    postMessage({ type: 'seg', i: 0, kind: 'text', text: tok.decode(out) });
    logits = step(next, true);
    if (n % 2 === 0) await yieldNow();
  }
  const raw = tok.decode(out);
  postMessage({ type: 'done', tokens: n, ms: performance.now() - t0, hitLimit: !ended && !stop, raw, ctx: cached.length });
}

onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'load') await load(m.url, m.version, m.size);
    else if (m.type === 'stop') stop = true;
    else if (m.type === 'reset') { core.ll_reset(); cached = []; }
    else if (m.type === 'chat') await answer(promptIds(m.opts.memory ? m.turns : [], m.text), m.opts);
    else if (m.type === 'regenerate') { if (!lastPrompt) throw new Error('nothing to regenerate'); await answer(lastPrompt, m.opts); }
  } catch (e) { postMessage({ type: 'error', message: e.message }); }
};
