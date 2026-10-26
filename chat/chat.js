'use strict';
// Multi-turn chat driver for N-Q-26M-Chat (runtime: nq_model.js, tokenizer: nq_tokenizer.js, tools: nq_circuits.js).
// The decode state (Gated DeltaNet matrices, attention K/V, n-gram history) lives across turns, so each turn only feeds its
// own tokens. Chat format: <bos> <USER>text<EOT><MODEL>[<THINK>…<ETHINK>]answer<EOT> <USER>…
// Tool calls: when the model writes <CALC>expr<EQ>, the executor forces the exact result + <ECALC>; shown as a chip.
const Lib = window.QOmniLib;
const $ = (id) => document.getElementById(id);
const ID = Lib.NQ_IDS;
let model = null, tok = null, generating = false, stopFlag = false, history = null;

function log(msg) { $('status').textContent = msg; }

async function fetchWithProgress(url, onProgress, timeoutMs = 120000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new Error(`timed out after ${timeoutMs / 1000}s`)), timeoutMs);
  let resp;
  try { resp = await fetch(url, { signal: ctrl.signal }); } finally { clearTimeout(timer); }
  if (!resp.ok) throw new Error(`fetch ${url} failed: ${resp.status}`);
  const total = Number(resp.headers.get('content-length')) || 0;
  if (!resp.body) return await resp.arrayBuffer();
  const reader = resp.body.getReader(); const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.byteLength; onProgress(got, total); }
  const out = new Uint8Array(got); let off = 0; for (const c of chunks) { out.set(c, off); off += c.byteLength; }
  return out.buffer;
}

async function loadModel() {
  const t0 = performance.now();
  const buf = await fetchWithProgress('./model.qpack?v=nq26m-2', (got, total) =>
    log(total ? `downloading model.qpack ${(got / 1e6).toFixed(1)}/${(total / 1e6).toFixed(1)} MB (${Math.round(100 * got / total)}%)…` : `downloading model.qpack ${(got / 1e6).toFixed(1)} MB…`));
  log(`unpacking (${(buf.byteLength / 1e6).toFixed(1)} MB in ${((performance.now() - t0) / 1000).toFixed(1)}s)…`);
  await new Promise((r) => setTimeout(r, 0));
  const c = Lib.readContainer(buf), loaded = Lib.loadNQ(c);
  tok = new Lib.NQTokenizer(c.json('tokenizer.json'));
  model = new Lib.NQ4JS(loaded);
  const cfg = loaded.cfg;
  $('params').textContent = `${cfg.num_hidden_layers} layers (${cfg.layer_pattern} ×${cfg.num_hidden_layers / cfg.layer_pattern.length}: Gated DeltaNet + gated attention) · hidden ${cfg.hidden_size} · vocab ${cfg.vocab_size.toLocaleString()}`;
  log('running self-test…'); await new Promise((r) => setTimeout(r, 0));
  const st = loaded.selftest; let ok = 0;
  if (st) for (const cs of st.cases) {
    model.reset(); let lg = null; cs.ids.forEach((id, i) => { lg = model.step(id, i === cs.ids.length - 1); });
    let a = 0; for (let v = 1; v < lg.length; v++) if (lg[v] > lg[a]) a = v;
    if (a === cs.top5[0] && Math.abs(lg[a] - cs.top5_logits[0]) < 5e-2) ok++;
  }
  if (st) $('selftest').textContent = `self-test: ${ok}/${st.cases.length} ${ok === st.cases.length ? '✓' : '✗ (results may be unreliable)'}`;
  $('backend').textContent = 'plain JavaScript (CPU, this tab)';
  resetConversation(); log('ready.'); $('send').disabled = false;
}

function resetConversation() {
  if (generating) stopFlag = true;
  model.reset(); history = [ID.BOS]; model.step(ID.BOS, false);
  $('log').innerHTML = '';
}

function addBubble(role, text) {
  const div = document.createElement('div'); div.className = `msg ${role}`; div.textContent = text;
  $('log').appendChild(div); $('log').scrollTop = $('log').scrollHeight; return div;
}

function pick(logits, greedy, recent) {
  const V = logits.length, l = new Float64Array(V);
  for (let i = 0; i < V; i++) l[i] = logits[i];
  for (let i = 8203; i < V; i++) l[i] = -Infinity;                // reserved control ids are never generated
  l[ID.USER] = l[ID.MODEL] = -Infinity;                              // the model never opens a turn itself
  if (greedy) { let b = 0; for (let i = 1; i < V; i++) if (l[i] > l[b]) b = i; return b; }
  for (const t of recent) if (t < 8192) l[t] = l[t] > 0 ? l[t] / 1.1 : l[t] * 1.1;   // repetition penalty 1.1 on recent text tokens
  const T = 0.7; let mx = -Infinity; for (let i = 0; i < V; i++) { l[i] /= T; if (l[i] > mx) mx = l[i]; }
  const idx = []; let sum = 0;
  for (let i = 0; i < V; i++) { const p = Math.exp(l[i] - mx); l[i] = p; sum += p; if (p > 1e-9) idx.push(i); }   // p(max) = 1
  idx.sort((a, b) => l[b] - l[a]);
  let cum = 0, cut = idx.length; for (let k = 0; k < idx.length; k++) { cum += l[idx[k]] / sum; if (cum >= 0.9) { cut = k + 1; break; } }   // top-p 0.9
  let r = Math.random() * idx.slice(0, cut).reduce((s, i) => s + l[i], 0);
  for (let k = 0; k < cut; k++) { r -= l[idx[k]]; if (r <= 0) return idx[k]; }
  return idx[0];
}

async function sendMessage() {
  if (generating) { stopFlag = true; return; }
  const text = $('input').value.trim(); if (!text) return;
  $('input').value = ''; generating = true; stopFlag = false; $('send').innerHTML = STOP_HTML;
  if (!$('memory').checked && history.length > 1) { model.reset(); history = [ID.BOS]; model.step(ID.BOS, false); }   // answer from a clean context
  addBubble('user', text);
  const think = $('think').checked, greedy = $('decoding').value === 'greedy';
  const turn = [ID.USER, ...tok.encode(text), ID.EOT, ID.MODEL, ...(think ? [ID.THINK] : [])];
  log('reading…'); await new Promise((r) => setTimeout(r, 0));
  let logits = null;
  turn.forEach((id, i) => { history.push(id); logits = model.step(id, i === turn.length - 1); });

  const bubble = addBubble('model', ''); bubble.classList.add('pending');
  let thinkBox = null, cur = document.createTextNode(''); bubble.appendChild(cur);
  const openThink = () => {
    const d = document.createElement('details'); d.className = 'think';
    d.innerHTML = '<summary><span data-lang="ru">размышления</span><span data-lang="en">thinking</span><span data-lang="zh">思考</span></summary>';
    bubble.insertBefore(d, cur); thinkBox = document.createElement('div'); d.appendChild(thinkBox);
    cur = document.createTextNode(''); thinkBox.appendChild(cur);
  };
  if (think) openThink();
  const ex = new Lib.CircuitExecutor(tok);
  let mode = 'text', expr = [], res = [], n = 0, ended = false; const t0 = performance.now();
  let tools = null;   // chips of calls made inside the (collapsed) thinking block stay visible in a row after it
  const chip = (e, r) => {
    const s = document.createElement('span'); s.className = 'tool-chip'; s.innerHTML = '<span class="tool-chip__tag">tool</span> ';
    s.appendChild(document.createTextNode(`${e} = ${r}`));
    if (thinkBox) {
      if (!tools) { tools = document.createElement('div'); tools.className = 'tools'; thinkBox.parentNode.after(tools); }
      tools.appendChild(s);
    } else { cur.parentNode.appendChild(s); cur = document.createTextNode(''); s.parentNode.appendChild(cur); }
  };
  while (n < 320 && !stopFlag) {
    let next = ex.next(history);
    if (next === null) next = pick(logits, greedy || mode === 'expr', history.slice(-64));   // tool expressions are always greedy
    history.push(next); n++;
    if (next === ID.EOT || next === ID.EOS) { ended = true; if (next === ID.EOS) history[history.length - 1] = ID.EOT; model.step(ID.EOT, false); break; }
    if (next === ID.THINK) openThink();
    else if (next === ID.ETHINK) { if (thinkBox) { cur = document.createTextNode(''); bubble.appendChild(cur); thinkBox = null; tools = null; } }
    else if (next === ID.CALC) { mode = 'expr'; expr = []; }
    else if (next === ID.EQ) { mode = 'result'; res = []; }
    else if (next === ID.ECALC) { mode = 'text'; chip(tok.decode(expr), tok.decode(res)); }
    else if (mode === 'expr') expr.push(next);
    else if (mode === 'result') res.push(next);
    else if (next < 8192) cur.nodeValue = tok.decode(curIds(cur, next));
    logits = model.step(next);
    if (n % 2 === 0) { $('tps').textContent = `${(n / ((performance.now() - t0) / 1000)).toFixed(1)} tok/s`; await new Promise((r) => setTimeout(r, 0)); $('log').scrollTop = $('log').scrollHeight; }
  }
  if (!ended) { history.push(ID.EOT); model.step(ID.EOT, false); }   // close the turn in the model state too
  bubble.classList.remove('pending');
  if (!bubble.textContent.trim()) bubble.textContent = '(no output)';
  else if (thinkBox && n >= 320) bubble.appendChild(document.createTextNode('(…the model was still thinking when the 320-token limit was reached — open "thinking")'));
  generating = false; $('send').innerHTML = SEND_HTML;
  log(`ready · ${n} tokens in ${((performance.now() - t0) / 1000).toFixed(1)}s · context ${history.length} tokens` + (history.length > 1500 ? ' — long conversation: start a new one for best results' : ''));
}

// text nodes decode their own token list (multi-byte UTF-8 characters can span several byte-level tokens)
const nodeIds = new WeakMap();
function curIds(node, id) { const a = nodeIds.get(node) || []; a.push(id); nodeIds.set(node, a); return a; }
const SEND_HTML = $('send').innerHTML;
const STOP_HTML = '<span data-lang="ru">Стоп</span><span data-lang="en">Stop</span><span data-lang="zh">停止</span>';

$('send').addEventListener('click', sendMessage);
$('input').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
$('newChat').addEventListener('click', resetConversation);
document.querySelectorAll('.try').forEach((el) => el.addEventListener('click', () => { $('input').value = el.textContent; $('input').focus(); }));
loadModel().catch((e) => { log('error: ' + e.message); console.error(e); });
