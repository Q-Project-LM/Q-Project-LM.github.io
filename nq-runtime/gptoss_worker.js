// Web Worker: gpt-oss (qpack container) on WebGPU, same chat protocol as nq_worker.js / llama_worker.js.
// Messages in:  {type:'load', url, version, size?}  {type:'chat', turns, text, opts}  {type:'regenerate', opts}  {type:'stop'}
// Messages out: progress {got, total | cached}, status, ready, seg {i, kind:'think'|'text', text}, done {tokens, ms, hitLimit, raw, ctx}, error
// The file (~8 GB) is streamed front to back straight into GPU buffers, and at the same time written to the origin-private file
// system (OPFS) so the next visit reads it from disk instead of the network. Harmony chat format: the model writes an "analysis"
// message (shown as thinking) and then a "final" one.
'use strict';
self.window = self;
const V = new URL(self.location.href).searchParams.get('v') || '1';
importScripts(`hf_tokenizer.js?v=${V}`, `gptoss_webgpu.js?v=${V}`);
const Lib = self.QPack;
const MAX_CTX = 2048;
let eng = null, tok = null, stop = false, cached = [], lastPrompt = null, SP = {};

async function opfsFile(name, create) {
  const root = await navigator.storage.getDirectory();
  return root.getFileHandle(name, { create });
}

async function load(url, version, size, nocache) {
  postMessage({ type: 'status', text: 'webgpu' });
  const { dev, info } = await Lib.GptOssWebGPU.device();
  eng = new Lib.GptOssWebGPU();
  eng.begin(dev, null);
  const name = 'qpack-' + (url.split('/').pop().split('?')[0]) + '-' + version;
  let fromCache = false;
  if (!nocache) try {   // complete cached copy?
    const mark = await (await (await opfsFile(name + '.ok', false)).getFile()).text();
    const fh = await opfsFile(name, false), file = await fh.getFile();
    if (Number(mark) === file.size && file.size > 0) {
      fromCache = true; postMessage({ type: 'progress', cached: true, got: 0, total: file.size });
      const CH = 64 << 20;
      for (let o = 0; o < file.size; o += CH) {
        eng.feed(new Uint8Array(await file.slice(o, Math.min(file.size, o + CH)).arrayBuffer()));
        postMessage({ type: 'progress', cached: true, got: Math.min(file.size, o + CH), total: file.size });
      }
    }
  } catch (e) { fromCache = false; }
  if (!fromCache) {
    let sync = null;
    if (!nocache) try {   // free the space of older versions, then write while downloading
      const root = await navigator.storage.getDirectory();
      for await (const [n] of root.entries()) if (n.startsWith('qpack-') && !n.startsWith(name)) await root.removeEntry(n);
      sync = await (await opfsFile(name, true)).createSyncAccessHandle(); sync.truncate(0);
    } catch (e) { sync = null; }
    const resp = await fetch(url); if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
    const total = Number(resp.headers.get('content-length')) || size || 0, rd = resp.body.getReader(); let got = 0, last = 0;
    for (;;) {
      const { done, value } = await rd.read(); if (done) break;
      if (sync) { try { sync.write(value, { at: got }); } catch (e) { try { sync.close(); } catch (_) {} sync = null; } }
      eng.feed(value); got += value.byteLength;
      if (got - last > (4 << 20) || got === total) { last = got; postMessage({ type: 'progress', got, total }); }
    }
    if (sync) { sync.flush(); sync.close(); try { const m = await (await opfsFile(name + '.ok', true)).createSyncAccessHandle(); const b = new TextEncoder().encode(String(got)); m.truncate(0); m.write(b, { at: 0 }); m.flush(); m.close(); } catch (e) {} }
  }
  if (!eng.done) throw new Error('the model file is incomplete');
  postMessage({ type: 'status', text: 'building' });
  tok = new Lib.HFTokenizer(JSON.parse(eng.text('tokenizer.json')));
  eng.init(MAX_CTX);
  for (const t of ['<|start|>', '<|end|>', '<|message|>', '<|channel|>', '<|return|>', '<|call|>']) SP[t] = tok.encode(t)[0];
  cached = []; lastPrompt = null;
  postMessage({ type: 'ready', selftest: [0, 0], decision: false, layers: eng.L, hidden: eng.d, vocab: eng.V, backend: 'webgpu',
                format: 'QPACK2 ' + (JSON.parse(eng.text('manifest.json')).quant || ''), arch: 'gpt-oss', gpu: (info.vendor || '') + ' ' + (info.architecture || ''),
                gpuMB: Math.round(eng.gpuBytes / 1048576) });
}

function promptIds(turns, text, think) {
  let s = '<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-08-05\n\nReasoning: ' +
    (think ? 'medium' : 'low') + '\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>';
  for (const t of turns) s += t.role === 'assistant' ? `<|start|>assistant<|channel|>final<|message|>${t.content}<|end|>` : `<|start|>user<|message|>${t.content}<|end|>`;
  s += `<|start|>user<|message|>${text}<|end|><|start|>assistant`;
  return tok.encode(s);
}

async function prefill(ids) {
  let k = 0; while (k < cached.length && k < ids.length - 1 && cached[k] === ids[k]) k++;
  eng.setPos(k); cached = cached.slice(0, k);
  let lg = null;
  for (let i = k; i < ids.length; i++) {
    lg = await eng.step(ids[i], i === ids.length - 1); cached.push(ids[i]);
    if (stop) throw new Error('stopped');
  }
  return lg;
}

const scratch = { idx: new Int32Array(64), val: new Float64Array(64) };
function pick(l, o, recent) {
  const Vn = l.length;
  for (const t of new Set(recent)) if (t < 199998) l[t] = l[t] > 0 ? l[t] / o.rep : l[t] * o.rep;
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
  let logits = await prefill(prompt);
  const t0 = performance.now();
  // harmony: <|channel|>NAME<|message|>BODY<|end|><|start|>assistant<|channel|>final<|message|>BODY<|return|>
  let mode = 0, chan = '', segI = -1, segIds = [], segKind = null, finalIds = [], n = 0, ended = false; const out = [];
  const open = (kind) => { segI++; segIds = []; segKind = kind; };
  while (n < o.max_new && !stop) {
    const next = pick(logits, o, out.slice(-64)); n++; out.push(next); cached.push(next);
    if (next === SP['<|return|>'] || next === SP['<|call|>']) { ended = true; break; }
    if (next === SP['<|channel|>']) { mode = 1; chan = ''; }
    else if (next === SP['<|message|>']) { mode = 2; open(chan.trim() === 'final' ? 'text' : 'think'); }
    else if (next === SP['<|end|>'] || next === SP['<|start|>']) { mode = 0; }
    else if (next >= 199998) { /* other special tokens: ignore */ }
    else if (mode === 1) chan += tok.decode([next]);
    else if (mode === 2) {
      segIds.push(next); if (segKind === 'text') finalIds.push(next);
      postMessage({ type: 'seg', i: segI, kind: segKind, text: tok.decode(segIds) });
    }
    logits = await eng.step(next, true);
  }
  const raw = tok.decode(finalIds);
  postMessage({ type: 'done', tokens: n, ms: performance.now() - t0, hitLimit: !ended && !stop, raw, ctx: cached.length });
}

let busy = Promise.resolve();
onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'stop') { stop = true; return; }
  busy = busy.then(async () => {
    try {
      if (m.type === 'load') await load(m.url, m.version, m.size, m.nocache);
      else if (m.type === 'reset') { eng.reset(); cached = []; }
      else if (m.type === 'logits') {   // debug: top-5 of the last position for given ids, from a clean state (compare with dump_logits)
        eng.reset(); cached = []; let lg = null; const t0 = performance.now();
        for (let i = 0; i < m.ids.length; i++) lg = await eng.step(m.ids[i], i === m.ids.length - 1);
        const ord = Array.from(lg.keys()).sort((a, b) => lg[b] - lg[a]).slice(0, 5);
        postMessage({ type: 'logits', top5: ord.map((i) => [i, lg[i]]), ms: performance.now() - t0 }); eng.reset();
      }
      else if (m.type === 'chat') await answer(promptIds(m.opts.memory ? m.turns : [], m.text, m.opts.think), m.opts);
      else if (m.type === 'regenerate') { if (!lastPrompt) throw new Error('nothing to regenerate'); await answer(lastPrompt, m.opts); }
    } catch (e) { postMessage({ type: 'error', message: e.message === 'stopped' ? 'stopped' : e.message }); }
  });
};
