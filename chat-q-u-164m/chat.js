'use strict';
// Multi-turn chat driver for Q-164M. Same loading/kernel plumbing as run/app.js (qpack.js/model.js/
// wasm.js are architecture-config-driven and reusable as-is), but the generation loop keeps the
// conversation's KV cache alive across turns instead of resetting per prompt, and renders a scrolling
// message list instead of a single output box. `circuits.js` here is the Q-architecture renumbering
// (see its own header comment) -- it must match whatever `model.qpack` this page is pointed at.
const Lib = window.QOmniLib;
const $ = (id) => document.getElementById(id);
let loaded = null, jsModel = null, wasmModel = null, tok = null, backend = 'none';
let generating = false, stopFlag = false;
let history = null; // token ids for the whole conversation so far (including the leading <bos>)

function log(msg) { $('status').textContent = msg; }

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
    const pct = total ? Math.min(100, Math.round(100 * got / total)) : null;
    log(total ? `${label} ${mb}/${(total / 1e6).toFixed(1)} MB (${pct}%)…` : `${label} ${mb} MB…`);
  };
  log('downloading model.qpack…');
  const [buf, wasmBytes] = await Promise.all([
    fetchWithProgress('./model.qpack', progress('downloading model.qpack')),
    fetchWithProgress('./kernel.wasm', progress('downloading kernel.wasm')).catch((e) => { console.warn('kernel.wasm fetch failed:', e); return null; }),
  ]);
  log(`parsing container (${(buf.byteLength / 1e6).toFixed(1)} MB downloaded in ${((performance.now() - t0) / 1000).toFixed(1)}s)…`);
  const c = Lib.readContainer(buf);
  loaded = Lib.loadTensors(c);
  tok = new Lib.ByteLevelBPE(c.json('tokenizer.json'));
  jsModel = new Lib.QOmniJS(loaded);
  $('params').textContent = `${loaded.cfg.num_hidden_layers} layers · hidden ${loaded.cfg.hidden_size} · vocab ${loaded.cfg.vocab_size.toLocaleString()}`;

  log("running self-test…");
  const st = loaded.selftest; let ok = 0;
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

  if (wasmBytes) {
    try { log('compiling the WebAssembly kernel and uploading weights…'); wasmModel = await Lib.loadWasmModel(loaded, wasmBytes); backend = 'wasm'; }
    catch (e) { console.warn('WASM init failed:', e); }
  }
  if (backend === 'none') backend = 'js';
  $('backend').textContent = backend === 'wasm' ? 'WebAssembly + SIMD' : 'plain JS fallback (slow)';
  resetConversation();
  log('ready.');
  $('send').disabled = false;
}

function activeModel() { return backend === 'wasm' ? wasmModel : jsModel; }

function resetConversation() {
  activeModel().reset();
  history = [1]; // <bos>; USER/MODEL/EOT turn markers get appended per message
  $('log').innerHTML = '';
}

function addBubble(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = text;
  $('log').appendChild(div);
  $('log').scrollIntoView({ block: 'end' });
  window.scrollTo(0, document.body.scrollHeight);
  return div;
}

function suppressImageAudio(logits, imageOffset) {
  if (imageOffset != null) for (let i = imageOffset; i < logits.length; i++) logits[i] = -Infinity;
  return logits;
}

// Exact circuits work under GREEDY decoding only (see run/app.js's Ask mode for why) -- chat mode
// always decodes greedily for the same reason: this is the methodology the published accuracy
// figures were measured with, and sampling makes the model much more likely to botch a <CALC> span.
function greedy(logits) {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; bi = i; }
  return bi;
}

async function feedTokens(ids) {
  let logits;
  for (const id of ids) logits = await activeModel().step(id);
  return logits;
}

async function sendMessage() {
  if (generating) { stopFlag = true; return; }
  const text = $('input').value.trim();
  if (!text) return;
  $('input').value = '';
  generating = true; stopFlag = false; $('send').textContent = 'Stop'; $('send').disabled = false;

  addBubble('user', text);
  const turnIds = [Lib.USER, ...tok.encode(text), Lib.EOT, Lib.MODEL];
  history.push(...turnIds);
  let logits = await feedTokens(turnIds);

  const bubble = addBubble('model', '');
  bubble.classList.add('pending');
  let cur = document.createTextNode('');
  bubble.appendChild(cur);
  const circuit = new Lib.CircuitInterceptor(tok);
  const maxNew = 200;
  // Unlike the single-shot run/ demo, this UI shows tool calls as a chip instead of hiding them --
  // "agentic" transparency: you see when the network hands off to the deterministic executor.
  let mode = 'text', exprChars = [], resultChars = [], n = 0;
  const t0 = performance.now();

  function appendChip(expr, result) {
    const chip = document.createElement('span');
    chip.className = 'tool-chip';
    chip.innerHTML = `<span class="tool-chip__tag">tool</span> ${expr.replace(/</g, '&lt;')} = ${result.replace(/</g, '&lt;')}`;
    bubble.appendChild(chip);
    cur = document.createTextNode('');
    bubble.appendChild(cur);
  }

  while (n < maxNew && !stopFlag) {
    let next = circuit.next();
    if (next === null) next = greedy(suppressImageAudio(logits.slice(), loaded.cfg.image_offset));
    if (next === 2 /* <eos> */) break;
    history.push(next);
    circuit.afterToken(history);
    if (next === Lib.CALC) { mode = 'expr'; exprChars = []; }
    else if (next === Lib.EQ) { mode = 'result'; resultChars = []; }
    else if (next === Lib.ECALC) { mode = 'text'; appendChip(exprChars.join(''), resultChars.join('')); }
    else if (mode === 'expr') exprChars.push(tok.decode([next]));
    else if (mode === 'result') resultChars.push(tok.decode([next]));
    else if (next < loaded.cfg.special_offset) cur.nodeValue += tok.decode([next]);
    if (next === Lib.EOT) break;
    n++;
    logits = await activeModel().step(next);
    await new Promise((r) => setTimeout(r, 0));
  }
  bubble.classList.remove('pending');
  if (!bubble.textContent) bubble.textContent = '(no output)';
  generating = false; $('send').textContent = 'Send';
  log(`ready · ${((performance.now() - t0) / 1000).toFixed(1)}s for this turn (${backend})`);
}

$('send').addEventListener('click', sendMessage);
$('input').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
$('newChat').addEventListener('click', resetConversation);
loadModel().catch((e) => { log('error: ' + e.message); console.error(e); });
