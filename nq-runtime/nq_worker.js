// Web Worker hosting an N-Q model (chat or JEV), so generation never blocks the page.
// Messages in:  {type:'load', url, version}
//               {type:'chat', turns:[{role,content}], text, opts:{think, memory, greedy, temperature, top_p, max_new, rep}}
//               (assistant turns carry the `raw` text returned in 'done', control tokens included)
//               {type:'regenerate', opts}   {type:'stop'}   {type:'reset'}
//               {type:'decide', state, questions, symmetric}
// Messages out: {type:'progress'|'ready'|'error'|'status'}, chat stream {type:'seg', i, kind:'text'|'think', text},
//               {type:'tool', expr, result, inThink}, {type:'done', tokens, ms, ctx}, decisions {type:'decision', result, ms}
'use strict';
self.window = self;   // the shared libraries attach to window.QOmniLib
const V = new URL(self.location.href).searchParams.get('v') || '1';
importScripts(`qpack.js?v=${V}`, `tokenizer.js?v=${V}`, `nq_tokenizer.js?v=${V}`, `nq_model.js?v=${V}`, `nq_circuits.js?v=${V}`, `nq_decide.js?v=${V}`);
const Lib = self.QOmniLib, ID = { BOS: 1, EOS: 2, USER: 8192, MODEL: 8193, EOT: 8194, THINK: 8195, ETHINK: 8196, CALC: 8197, EQ: 8198, ECALC: 8199 };
let model = null, tok = null, stop = false, busy = false;
let history = null;            // token ids currently inside the model state
let synced = [];               // the [{role, content}] turns that `history` encodes
let beforeAnswer = null;       // snapshot right before the last answer (for regenerate)

async function fetchCached(url, version) {
  const key = `${url}#${version}`;
  try {
    const cache = await caches.open('nq-models');
    const hit = await cache.match(key);
    if (hit) { postMessage({ type: 'progress', cached: true }); return await hit.arrayBuffer(); }
    const resp = await fetch(url); if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
    const total = Number(resp.headers.get('content-length')) || 0, reader = resp.body.getReader(), chunks = []; let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.byteLength; postMessage({ type: 'progress', got, total }); }
    const buf = new Uint8Array(got); let off = 0; for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
    try { for (const k of await cache.keys()) if (k.url.split('#')[0].endsWith(url.split('?')[0])) await cache.delete(k); await cache.put(key, new Response(buf)); } catch (e) {}
    return buf.buffer;
  } catch (e) {
    if (String(e.message).startsWith('fetch')) throw e;
    const resp = await fetch(url); return await resp.arrayBuffer();      // Cache API unavailable (e.g. private mode)
  }
}

async function load(url, version) {
  const buf = await fetchCached(url, version);
  postMessage({ type: 'status', text: 'unpacking' });
  const c = Lib.readContainer(buf), loaded = Lib.loadNQ(c);
  tok = new Lib.NQTokenizer(c.json('tokenizer.json')); model = new Lib.NQ4JS(loaded);
  let ok = 0, n = 0;
  if (loaded.selftest) for (const cs of loaded.selftest.cases) {
    n++; model.reset(); let lg = null; cs.ids.forEach((id, i) => { lg = model.step(id, i === cs.ids.length - 1); });
    let a = 0; for (let v = 1; v < lg.length; v++) if (lg[v] > lg[a]) a = v;
    ok += a === cs.top5[0] && Math.abs(lg[a] - cs.top5_logits[0]) < 5e-2;
  }
  resetChat();
  const cfg = loaded.cfg;
  postMessage({ type: 'ready', selftest: [ok, n], decision: !!loaded.tensors['decide_head.weight'], layers: cfg.num_hidden_layers, hidden: cfg.hidden_size, vocab: cfg.vocab_size });
}

function resetChat() { model.reset(); history = [ID.BOS]; model.step(ID.BOS, false); synced = []; beforeAnswer = null; }
const sameTurns = (a, b) => a.length === b.length && a.every((t, i) => t.role === b[i].role && t.content === b[i].content);

// bring the model state to exactly `turns` (cheap when only appending; a full re-read after switching conversations)
function syncTo(turns) {
  if (!(synced.length <= turns.length && sameTurns(synced, turns.slice(0, synced.length)))) resetChat();
  for (const t of turns.slice(synced.length)) {
    const ids = t.role === 'user' ? [ID.USER, ...tok.encode(t.content), ID.EOT] : [ID.MODEL, ...tok.encode(t.content), ID.EOT];   // assistant content = raw (control tokens kept)
    for (const id of ids) { history.push(id); model.step(id, false); }
    synced.push(t);
  }
}

// forced = greedy without the repetition penalty (thinking / tool expressions, where repeated numbers are legitimate)
function pick(logits, o, forced, recent) {
  const V = logits.length, l = new Float64Array(V);
  for (let i = 0; i < V; i++) l[i] = logits[i];
  for (let i = 8203; i < V; i++) l[i] = -Infinity;
  l[ID.USER] = l[ID.MODEL] = -Infinity;
  const argmax = () => { let b = 0; for (let i = 1; i < V; i++) if (l[i] > l[b]) b = i; return b; };
  if (forced) return argmax();
  for (const t of new Set(recent)) if (t < 8192) l[t] = l[t] > 0 ? l[t] / o.rep : l[t] * o.rep;
  if (o.greedy) return argmax();
  let mx = -Infinity; for (let i = 0; i < V; i++) { l[i] /= o.temperature; if (l[i] > mx) mx = l[i]; }
  const idx = []; for (let i = 0; i < V; i++) { l[i] = Math.exp(l[i] - mx); if (l[i] > 1e-9) idx.push(i); }
  idx.sort((a, b) => l[b] - l[a]); const sum = idx.reduce((s, i) => s + l[i], 0);
  let cum = 0, cut = idx.length; for (let k = 0; k < idx.length; k++) { cum += l[idx[k]] / sum; if (cum >= o.top_p) { cut = k + 1; break; } }
  let r = Math.random() * idx.slice(0, cut).reduce((s, i) => s + l[i], 0);
  for (let k = 0; k < cut; k++) { r -= l[idx[k]]; if (r <= 0) return idx[k]; }
  return idx[0];
}

async function generate(o) {
  // answer() pushed the prompt's last token into `history` without feeding it: its step gives the first logits
  const t0 = performance.now(), start = history.lastIndexOf(ID.MODEL) + 1;   // raw includes a forced <THINK>
  let logits = model.step(history[history.length - 1], true);
  const ex = new Lib.CircuitExecutor(tok); let seg = { i: 0, kind: 'text', ids: [] }, mode = 'text', expr = [], res = [], n = 0, inThink = false, thinkLen = 0;
  const flush = () => postMessage({ type: 'seg', i: seg.i, kind: seg.kind, text: tok.decode(seg.ids) });
  const newSeg = (kind) => { seg = { i: seg.i + 1, kind, ids: [] }; };
  if (history[history.length - 1] === ID.THINK) { inThink = true; newSeg('think'); }
  let ended = false;
  while (n < o.max_new && !stop) {
    let next = ex.next(history);
    // thinking budget: a 26M model can ramble inside <THINK> forever; past the budget (and not inside a tool call) close it
    if (next === null && inThink && mode === 'text' && ++thinkLen > (o.think_budget || 128)) next = ID.ETHINK;
    // greedy inside <THINK> and tool expressions (how the model was evaluated); sampling only in the visible answer
    if (next === null) next = pick(logits, o, mode === 'expr' || inThink, history.slice(-64));
    history.push(next); n++;
    if (next === ID.EOT || next === ID.EOS) { ended = true; history[history.length - 1] = ID.EOT; model.step(ID.EOT, false); break; }
    if (next === ID.THINK) { inThink = true; thinkLen = 0; newSeg('think'); }
    else if (next === ID.ETHINK) { inThink = false; newSeg('text'); }
    else if (next === ID.CALC) { mode = 'expr'; expr = []; }
    else if (next === ID.EQ) { mode = 'result'; res = []; }
    else if (next === ID.ECALC) { mode = 'text'; postMessage({ type: 'tool', expr: tok.decode(expr), result: tok.decode(res), inThink }); newSeg(inThink ? 'think' : 'text'); }
    else if (mode === 'expr') expr.push(next);
    else if (mode === 'result') res.push(next);
    else if (next < 8192) { seg.ids.push(next); flush(); }
    logits = model.step(next, true);
    if (n % 3 === 0) await new Promise((r) => setTimeout(r, 0));   // lets 'stop' messages in
  }
  if (!ended) { history.push(ID.EOT); model.step(ID.EOT, false); }
  // raw = the assistant turn exactly as the model state holds it (control tokens included), re-encodable by syncTo()
  const raw = tok.decode(history.slice(start, -1), false);
  synced = synced.concat([{ role: 'assistant', content: raw }]);
  return { tokens: n, ms: performance.now() - t0, hitLimit: !ended && !stop, raw };
}

async function answer(turns, text, o, regenerate) {
  busy = true; stop = false;
  try {
    if (!regenerate) {
      if (!o.memory) turns = [];
      syncTo(turns);
      const prompt = [ID.USER, ...tok.encode(text), ID.EOT, ID.MODEL, ...(o.think ? [ID.THINK] : [])];
      for (const id of prompt.slice(0, -1)) { history.push(id); model.step(id, false); }
      history.push(prompt[prompt.length - 1]);            // its step happens at the start of generate()
      synced = synced.concat([{ role: 'user', content: text }]);
      beforeAnswer = { snap: model.snapshot(), hist: history.slice(), synced: synced.slice() };
    } else {
      if (!beforeAnswer) throw new Error('nothing to regenerate');
      model.restore(beforeAnswer.snap); history = beforeAnswer.hist.slice(); synced = beforeAnswer.synced.slice();
    }
    const r = await generate(o);
    postMessage({ type: 'done', ...r, ctx: history.length });
  } catch (e) { postMessage({ type: 'error', message: e.message }); }
  busy = false;
}

onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'load') await load(m.url, m.version);
    else if (m.type === 'stop') stop = true;
    else if (m.type === 'reset') resetChat();
    else if (m.type === 'chat') await answer(m.turns, m.text, m.opts, false);
    else if (m.type === 'regenerate') await answer(null, null, m.opts, true);
    else if (m.type === 'decide') {
      const t0 = performance.now();
      const result = Lib.decide(model, tok, m.state, m.questions, { symmetric: m.symmetric !== false, onProgress: (k) => postMessage({ type: 'status', text: `${k} tokens` }) });
      postMessage({ type: 'decision', result, ms: performance.now() - t0 });
    }
  } catch (e) { postMessage({ type: 'error', message: e.message }); }
};
