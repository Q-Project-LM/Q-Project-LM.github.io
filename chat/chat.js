'use strict';
// N-Q-26M-Chat page: UI only. The model runs in ../nq-runtime/nq_worker.js (Web Worker); conversations live in localStorage.
// A conversation: {id, title, updated, messages: [{role:'user', content} | {role:'assistant', raw, segs:[{kind, text}], tools:[...], meta}]}
// `raw` is the assistant turn with its control tokens (<THINK>, <CALC>…) — that is what the model re-reads.
(function () {
  const V = 'nq-12';   // code version
  // selectable models: the small one is served by the site, the 164M one (230 MB, over GitHub's file limit) from Hugging Face
  const MODELS = {
    nq26m: { name: 'N-Q-26M-Chat', url: 'model.q8.qpack', v: 'nq26m-q8-1', mb: 47, blog: '../blog-nq26m.html', size: ['26M параметров', '26M parameters', '26M 参数'] },
    nq164m: { name: 'N-Q-164M-Chat', url: 'https://huggingface.co/q-project/N-Q-164M-Chat/resolve/main/model.q8.qpack', v: 'nq164m-q8-1', mb: 230, blog: '../blog-nq164m.html', size: ['164M параметров', '164M parameters', '164M 参数'] },
    qwen05: { name: 'Qwen2.5-0.5B-Instruct · qpack q4', url: 'https://huggingface.co/q-project/Qwen2.5-0.5B-Instruct-qpack/resolve/main/qwen2.5-0.5b-instruct.trim80k.q4.qpack', v: 'qwen05-trim80k-q4-1', mb: 269,
      worker: 'llama_worker.js', blog: '../blog-qpack.html', size: ['Qwen2.5 0.5B (Alibaba), 4 бита', 'Qwen2.5 0.5B (Alibaba), 4-bit', 'Qwen2.5 0.5B（阿里巴巴），4 比特'] },
    gptoss: { name: 'gpt-oss-20b · qpack q1', url: 'https://huggingface.co/q-project/gpt-oss-20b-qpack-q1/resolve/main/gptoss-20b.c7-mixed.qpack', v: 'gptoss-c7-1', mb: 8209,
      worker: 'gptoss_worker.js', gate: true, blog: '../blog-qpack.html', size: ['gpt-oss-20b (OpenAI), 1-битные эксперты, WebGPU', 'gpt-oss-20b (OpenAI), 1-bit experts, WebGPU', 'gpt-oss-20b（OpenAI），1 比特专家，WebGPU'] },
  };
  let modelKey = 'nq26m';
  try { const k = localStorage.getItem('nq-chat-model'); if (MODELS[k]) modelKey = k; } catch (e) {}
  try { const k = new URLSearchParams(location.search).get('model'); if (MODELS[k]) modelKey = k; } catch (e) {}
  const $ = (id) => document.getElementById(id);
  const lang = () => (document.documentElement.lang || 'en').slice(0, 2);
  // a trilingual label that follows the site's language switch (main.js toggles .is-active on [data-lang])
  const L = (ru, en, zh) => ['ru', 'en', 'zh'].map((l, i) => `<span data-lang="${l}"${lang() === l ? ' class="is-active"' : ''}>${esc([ru, en, zh || en][i])}</span>`).join('');
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const SUGGEST = [
    ['Сколько будет 4821 умножить на 37?', 'точный калькулятор', 'exact calculator'],
    ['Какой день недели 9 мая 2027 года?', 'даты', 'dates'],
    ['Привет! Кто ты?', 'знакомство', 'say hi'],
    ['Convert 100 kilograms to pounds.', 'units', 'units'],
    ['Отсортируй по возрастанию: 42, 7, 19, 3.', 'сортировка', 'sorting'],
    ['Напиши короткое стихотворение про осень.', 'текст', 'writing'],
  ];

  // ---------------------------------------------------------------- storage
  const KEY = 'nq-chat-convs';
  let convs = [], cur = null;
  function loadConvs() { try { convs = JSON.parse(localStorage.getItem(KEY) || '[]'); } catch (e) { convs = []; } }
  function saveConvs() { try { localStorage.setItem(KEY, JSON.stringify(convs.slice(0, 50))); } catch (e) {} }
  function newConv() { cur = { id: Date.now().toString(36), title: '', updated: Date.now(), messages: [] }; renderAll(); }
  function touch() {
    cur.updated = Date.now();
    if (!cur.title) { const u = cur.messages.find((m) => m.role === 'user'); if (u) cur.title = u.content.slice(0, 60); }
    if (!convs.includes(cur)) convs.unshift(cur);
    convs.sort((a, b) => b.updated - a.updated); saveConvs(); renderList();
  }

  // ---------------------------------------------------------------- markdown (escaped first; a small safe subset)
  function md(src) {
    const out = []; const lines = esc(src).split('\n'); let i = 0;
    const inline = (s) => s.replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<i>$2</i>');
    while (i < lines.length) {
      const l = lines[i];
      if (/^```/.test(l)) { const buf = []; i++; while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]); i++; out.push(`<pre><code>${buf.join('\n')}</code></pre>`); continue; }
      const h = l.match(/^(#{1,4})\s+(.*)/); if (h) { out.push(`<h4>${inline(h[2])}</h4>`); i++; continue; }
      if (/^\s*([-*•]|\d+[.)])\s+/.test(l)) {
        const ordered = /^\s*\d+[.)]/.test(l), items = [];
        while (i < lines.length && /^\s*([-*•]|\d+[.)])\s+/.test(lines[i])) items.push(inline(lines[i++].replace(/^\s*([-*•]|\d+[.)])\s+/, '')));
        out.push(`<${ordered ? 'ol' : 'ul'}>${items.map((x) => `<li>${x}</li>`).join('')}</${ordered ? 'ol' : 'ul'}>`); continue;
      }
      if (!l.trim()) { i++; continue; }
      const para = []; while (i < lines.length && lines[i].trim() && !/^(```|#{1,4}\s|\s*([-*•]|\d+[.)])\s)/.test(lines[i])) para.push(inline(lines[i++]));
      out.push(`<p>${para.join('<br>')}</p>`);
    }
    return out.join('');
  }

  // ---------------------------------------------------------------- rendering
  const ICON = {
    copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>',
    redo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M20 12a8 8 0 1 1-2.3-5.6M20 4v5h-5"/></svg>',
  };
  function renderList() {
    $('convList').innerHTML = '';
    for (const c of convs) {
      const d = document.createElement('div'); d.className = 'conv' + (c === cur ? ' is-current' : '');
      d.innerHTML = `<span class="conv__title"></span><button class="conv__del" title="Delete" aria-label="Delete">×</button>`;
      d.querySelector('.conv__title').textContent = c.title || '…';
      d.addEventListener('click', (e) => {
        if (e.target.classList.contains('conv__del')) { convs = convs.filter((x) => x !== c); saveConvs(); if (c === cur) newConv(); else renderList(); return; }
        if (busy) return; cur = c; renderAll(); closeSide();
      });
      $('convList').appendChild(d);
    }
  }
  function botBody(m) {
    const parts = [];
    for (const s of m.segs || []) {
      if (s.kind === 'think') parts.push(`<details class="think"><summary>${L('размышления', 'thinking', '思考')}</summary><div class="md">${md(s.text)}</div></details>`);
      else if (s.kind === 'tools') parts.push(`<div class="tools">${s.items.map((t) => `<span class="chip${t.result === '?' ? ' is-fail' : ''}"><b>tool</b>${esc(t.expr)} = ${esc(t.result)}</span>`).join('')}</div>`);
      else if (s.text.trim()) parts.push(`<div class="md">${md(s.text)}</div>`);
    }
    if (m.note) parts.push(`<div class="m__note">${m.note}</div>`);
    return parts.join('') || `<div class="md"></div>`;
  }
  function msgEl(m, idx) {
    const d = document.createElement('div');
    if (m.role === 'user') { d.className = 'm m--user'; d.innerHTML = '<div class="m__body"></div>'; d.firstChild.textContent = m.content; return d; }
    d.className = 'm m--bot';
    d.innerHTML = `<div class="m__avatar"><img src="../assets/q-logo.svg" alt=""></div><div class="m__body"><div class="m__content">${botBody(m)}</div><div class="m__meta"></div></div>`;
    const meta = d.querySelector('.m__meta');
    if (m.meta) meta.insertAdjacentHTML('beforeend', `<span>${m.meta.tokens} tok · ${(m.meta.tokens / (m.meta.ms / 1000)).toFixed(1)} tok/s</span>`);
    const copy = document.createElement('button'); copy.innerHTML = ICON.copy + L('копировать', 'copy', '复制');
    copy.onclick = () => navigator.clipboard && navigator.clipboard.writeText((m.segs || []).filter((s) => s.kind === 'text').map((s) => s.text).join('\n').trim());
    meta.appendChild(copy);
    if (idx === cur.messages.length - 1) {
      const re = document.createElement('button'); re.innerHTML = ICON.redo + L('ещё раз', 'regenerate', '重新生成'); re.onclick = regenerate; meta.appendChild(re);
    }
    return d;
  }
  function renderAll() {
    $('msgs').innerHTML = ''; cur.messages.forEach((m, i) => $('msgs').appendChild(msgEl(m, i)));
    $('empty').hidden = !ready || cur.messages.length > 0; renderList(); scrollDown(true);
  }
  function scrollDown(force) { const t = $('thread'); if (force || t.scrollHeight - t.scrollTop - t.clientHeight < 160) t.scrollTop = t.scrollHeight; }

  // ---------------------------------------------------------------- worker
  let worker = null, ready = false, busy = false, live = null, lastAnswered = null, lastMemory = true;
  function applyModelUi() {
    const M = MODELS[modelKey], [sru, sen, szh] = M.size;
    $('modelSel').value = modelKey; $('modelName').textContent = M.name; $('emptyTitle').textContent = M.name; $('aboutLink').href = M.blog;
    $('loadHint').innerHTML = L(`${M.mb} МБ, один раз — дальше модель берётся из кеша браузера.`, `${M.mb} MB, once — after that the model loads from the browser cache.`, `${M.mb} MB，只需下载一次 —— 之后从浏览器缓存加载。`);
    $('disclaimer').innerHTML = L(`${sru}: часто путает факты. Работает локально, ничего не отправляется на сервер.`, `${sen}: often gets facts wrong. Runs locally — nothing is sent to a server.`, `${szh}：经常弄错事实。在本地运行 —— 不会向服务器发送任何内容。`);
    document.title = `Chat with ${M.name} — Q Project`;
  }
  // big models: check WebGPU / limits / storage first and download only after an explicit click
  async function gateCheck() {
    const items = []; let ok = true;
    const add = (good, ru, en) => { items.push([good, L(ru, en, en)]); if (good === false) ok = false; };
    let ad = null;
    try { ad = navigator.gpu ? await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }) : null; } catch (e) { ad = null; }
    if (!navigator.gpu) add(false, 'WebGPU недоступен в этом браузере (нужен Chrome / Edge 113+ на компьютере)', 'WebGPU is not available in this browser (Chrome / Edge 113+ on a desktop)');
    else if (!ad) add(false, 'WebGPU есть, но видеокарта не найдена', 'WebGPU is present but no GPU adapter was found');
    else {
      const i = ad.info || {}, mb = Math.round(ad.limits.maxBufferSize / 1048576), sb = Math.round(ad.limits.maxStorageBufferBindingSize / 1048576);
      add(true, `WebGPU: ${i.vendor || '?'} ${i.architecture || ''}`.trim(), `WebGPU: ${i.vendor || '?'} ${i.architecture || ''}`.trim());
      const big = Math.min(mb, sb) >= 600;
      add(big, `лимит буфера ${Math.min(mb, sb)} МБ (нужно ≥ 600)`, `buffer limit ${Math.min(mb, sb)} MB (≥ 600 needed)`);
      if (ad.isFallbackAdapter || i.architecture === 'swiftshader') add(null, 'это программный (CPU) адаптер — будет очень медленно', 'this is a software (CPU) adapter — it will be very slow');
    }
    try { const est = await navigator.storage.estimate(); const free = (est.quota - est.usage) / 1e9;
      add(free > 8.5 ? true : null, `место для кэша: ${free.toFixed(0)} ГБ${free > 8.5 ? '' : ' — не хватит, модель будет качаться заново при каждом заходе'}`, `cache space: ${free.toFixed(0)} GB${free > 8.5 ? '' : ' — not enough, the model will be downloaded on every visit'}`); } catch (e) {}
    return { ok, items };
  }
  async function showGate() {
    const g = $('gate'); g.hidden = false; $('loadText').textContent = ''; $('loadBar').parentElement.hidden = true;
    g.innerHTML = L('Проверяю браузер…', 'Checking the browser…', 'Checking the browser…');
    const r = await gateCheck();
    g.innerHTML = `<b>${L('gpt-oss-20b прямо в браузере', 'gpt-oss-20b right in the browser', 'gpt-oss-20b right in the browser')}</b>
      <div>${L('7.8 ГБ весов: 1-битные эксперты + точные MXFP4, ~6.6 ГБ видеопамяти и ~1.2 ГБ оперативной. Нужна видеокарта с 8 ГБ+ (или Apple Silicon с 16 ГБ). Скачивается один раз и хранится в кэше сайта. Качество заметно ниже оригинала (перплексия ×1.4–3.6).',
        '7.8 GB of weights: 1-bit experts + exact MXFP4 ones, ~6.6 GB of GPU memory and ~1.2 GB of RAM. Needs an 8 GB+ GPU (or Apple Silicon with 16 GB). Downloaded once and kept in the site cache. Quality is clearly below the original (perplexity ×1.4–3.6).',
        '7.8 GB of weights, ~6.6 GB GPU memory. Quality is below the original.')}</div>
      <ul>${r.items.map(([good, t]) => `<li class="${good === true ? 'ok' : good === false ? 'bad' : ''}">${t}</li>`).join('')}</ul>
      ${r.ok ? `<button class="btn btn--solid btn--sm" id="gateGo" type="button">${L('Скачать и запустить (7.8 ГБ)', 'Download and run (7.8 GB)', 'Download and run (7.8 GB)')}</button>` : ''}`;
    if (r.ok) $('gateGo').onclick = () => { g.hidden = true; $('loadBar').parentElement.hidden = false; startWorker(true); };
  }
  function startWorker(confirmed) {
    if (worker) { worker.terminate(); worker = null; }
    if (MODELS[modelKey].gate && !confirmed) {
      ready = false; busy = false; applyModelUi(); $('loading').hidden = false; $('empty').hidden = true; $('modelMeta').textContent = '…'; setStatus('loading'); updateSend();
      showGate(); return;
    }
    $('gate').hidden = true; $('loadBar').parentElement.hidden = false;
    ready = false; busy = false; live = null; lastAnswered = null;
    applyModelUi(); $('loading').hidden = false; $('empty').hidden = true; $('loadBar').style.width = '0%'; $('loadText').textContent = 'Downloading the model…';
    $('modelMeta').textContent = '…'; setStatus('loading'); updateSend();
    const M = MODELS[modelKey];
    worker = new Worker(`../nq-runtime/${M.worker || 'nq_worker.js'}?v=${V}`);
    worker.onmessage = onWorker;
    worker.postMessage({ type: 'load', url: new URL(`${M.url}${M.url.includes('?') ? '&' : '?'}v=${M.v}`, location.href).href, version: M.v, size: M.mb * 1e6 });   // absolute: the worker resolves URLs from /nq-runtime/
  }
  function onWorker(ev) {
    const m = ev.data;
    if (m.type === 'progress') {
      if (m.cached) { $('loadText').textContent = 'Loading from browser cache…' + (m.total ? ` ${(m.got / 1e6).toFixed(0)} / ${(m.total / 1e6).toFixed(0)} MB` : ''); $('loadBar').style.width = m.total ? Math.round(100 * m.got / m.total) + '%' : '100%'; }
      else { const pct = m.total ? Math.round(100 * m.got / m.total) : 0; $('loadBar').style.width = pct + '%'; $('loadText').textContent = `Downloading the model… ${(m.got / 1e6).toFixed(1)}${m.total ? ' / ' + (m.total / 1e6).toFixed(1) : ''} MB`; }
    } else if (m.type === 'status' && !ready) { $('loadText').textContent = ({ webgpu: 'Starting WebGPU…', building: 'Building GPU pipelines…' })[m.text] || 'Unpacking and self-testing…'; }
    else if (m.type === 'ready') {
      ready = true; $('loading').hidden = true; setStatus('ready');
      $('modelMeta').innerHTML = `${m.layers} layers · hidden ${m.hidden} · vocab ${m.vocab.toLocaleString()}<br>` + (m.backend === 'webgpu'
        ? `WebGPU · ${esc(m.gpu || '')} · ${m.gpuMB} MB on the GPU · ${esc(m.format)}`
        : `self-test ${m.selftest[0]}/${m.selftest[1]} ${m.selftest[0] === m.selftest[1] ? '✓' : '✗'} · ${m.format} · ${({ 'wasm-core': 'WASM SIMD core', wasm: 'WASM SIMD', js: 'JS' })[m.backend]}`);
      updateSend(); renderAll(); $('input').focus();
    } else if (m.type === 'seg' && live) {
      let s = live.segs.find((x) => x.i === m.i && x.kind === m.kind);
      if (!s) { s = { i: m.i, kind: m.kind, text: '' }; live.segs.push(s); }
      s.text = m.text; redrawLive();
    } else if (m.type === 'tool' && live) {
      const last = live.segs[live.segs.length - 1];
      if (last && last.kind === 'tools') last.items.push({ expr: m.expr, result: m.result }); else live.segs.push({ kind: 'tools', items: [{ expr: m.expr, result: m.result }] });
      redrawLive();
    } else if (m.type === 'done' && live) {
      live.raw = m.raw; live.meta = { tokens: m.tokens, ms: m.ms };
      if (m.hitLimit) live.note = L(`(ответ обрезан на ${opts().max_new} токенах — лимит в настройках)`, `(cut at ${opts().max_new} tokens — the limit is in settings)`, `（在 ${opts().max_new} 个词元处截断 —— 可在设置中调整上限）`);
      if (!live.segs.some((s) => (s.kind === 'text' && s.text.trim()) || s.kind === 'tools')) live.note = (live.note || '') + L(' (только размышления, без ответа)', ' (only thinking, no answer)', '（只有思考，没有回答）');
      finishLive(); setStatus('ready');
    } else if (m.type === 'error') {
      if (!ready) { $('loadText').textContent = 'Error: ' + m.message; }
      else if (live) { live.note = 'error: ' + esc(m.message); finishLive(); }
      setStatus(ready ? 'ready' : 'error');
    }
  }

  function setStatus(s) {
    const p = $('statusPill'); p.classList.toggle('is-ready', s === 'ready'); p.classList.toggle('is-busy', s === 'busy');
    $('statusText').textContent = { ready: 'ready', busy: 'generating', loading: 'loading', error: 'error' }[s] || s;
  }
  function redrawLive() {
    const el = $('msgs').lastElementChild; if (!el) return;
    el.querySelector('.m__content').innerHTML = botBody(live); scrollDown(false);
  }
  function finishLive() {
    busy = false; lastAnswered = cur; lastMemory = opts().memory; const idx = cur.messages.length - 1; cur.messages[idx] = live; live = null;
    touch(); renderAll(); updateSend();   // re-render: only the newest answer gets the regenerate button
  }

  // ---------------------------------------------------------------- sending
  const opts = () => ({ temperature: +$('temp').value, top_p: +$('topP').value, max_new: +$('maxNew').value, rep: +$('rep').value,
    greedy: $('greedy').checked, think: $('think').checked, memory: $('memory').checked });
  const turnsOf = (msgs) => msgs.filter((m) => m.role === 'user' || m.raw != null).map((m) => ({ role: m.role, content: m.role === 'user' ? m.content : m.raw }));

  function startLive() {
    live = { role: 'assistant', segs: [], raw: null }; cur.messages.push(live);
    const el = msgEl(live, cur.messages.length - 1); el.classList.add('is-pending'); el.querySelector('.m__meta').innerHTML = ''; $('msgs').appendChild(el);
    busy = true; setStatus('busy'); updateSend(); scrollDown(true);
  }
  function send(text) {
    text = (text || '').trim(); if (!text || !ready || busy) return;
    const prior = turnsOf(cur.messages);
    cur.messages.push({ role: 'user', content: text }); $('empty').hidden = true;
    $('msgs').appendChild(msgEl(cur.messages[cur.messages.length - 1])); touch();
    startLive(); worker.postMessage({ type: 'chat', turns: prior, text, opts: opts() });
  }
  function regenerate() {
    if (busy || !cur.messages.length || cur.messages[cur.messages.length - 1].role !== 'assistant') return;
    const prior = turnsOf(cur.messages.slice(0, -2)), text = cur.messages[cur.messages.length - 2].content;
    cur.messages.pop(); $('msgs').removeChild($('msgs').lastElementChild);
    startLive();
    // the worker keeps a snapshot from right before the last answer it generated: reuse it when that answer is this one
    if (lastAnswered === cur && opts().memory === lastMemory) worker.postMessage({ type: 'regenerate', opts: opts() });
    else worker.postMessage({ type: 'chat', turns: prior, text, opts: opts() });
  }
  function updateSend() {
    const b = $('send'); b.classList.toggle('is-stop', busy); b.disabled = !ready || (!busy && !$('input').value.trim());
  }

  // ---------------------------------------------------------------- wiring
  $('composer').addEventListener('submit', (e) => { e.preventDefault(); if (busy) { worker.postMessage({ type: 'stop' }); return; } const t = $('input').value; $('input').value = ''; autoGrow(); updateSend(); send(t); });
  const autoGrow = () => { const t = $('input'); t.style.height = 'auto'; t.style.height = Math.min(200, t.scrollHeight) + 'px'; };
  $('input').addEventListener('input', () => { autoGrow(); updateSend(); });
  $('input').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('composer').requestSubmit(); } });
  $('newChat').addEventListener('click', () => { if (!busy) { newConv(); closeSide(); $('input').focus(); } });
  const closeSide = () => $('app').classList.remove('side-open');
  $('menuBtn').addEventListener('click', () => $('app').classList.toggle('side-open'));
  $('scrim').addEventListener('click', closeSide);
  $('settingsBtn').addEventListener('click', (e) => { e.stopPropagation(); const s = $('settings'); s.hidden = !s.hidden; $('settingsBtn').setAttribute('aria-expanded', String(!s.hidden)); });
  document.addEventListener('click', (e) => { if (!$('settings').hidden && !$('settings').contains(e.target)) $('settings').hidden = true; });
  for (const [id, out, d] of [['temp', 'tempVal', 2], ['topP', 'topPVal', 2], ['maxNew', 'maxNewVal', 0], ['rep', 'repVal', 2]]) {
    $(id).addEventListener('input', () => { $(out).textContent = (+$(id).value).toFixed(d); savePrefs(); });
  }
  ['greedy', 'think', 'memory'].forEach((id) => $(id).addEventListener('change', savePrefs));
  function savePrefs() { try { localStorage.setItem('nq-chat-prefs-v3', JSON.stringify(opts())); } catch (e) {} }
  try {
    const p = JSON.parse(localStorage.getItem('nq-chat-prefs-v3') || 'null');
    if (p) { $('temp').value = p.temperature; $('topP').value = p.top_p; $('maxNew').value = p.max_new; $('rep').value = p.rep; $('greedy').checked = p.greedy; $('think').checked = p.think; $('memory').checked = p.memory;
      for (const [id, out, d] of [['temp', 'tempVal', 2], ['topP', 'topPVal', 2], ['maxNew', 'maxNewVal', 0], ['rep', 'repVal', 2]]) $(out).textContent = (+$(id).value).toFixed(d); }
  } catch (e) {}
  for (const [q, ru, en] of SUGGEST) {
    const b = document.createElement('button'); b.type = 'button'; b.innerHTML = `${esc(q)}<small>${L(ru, en, en)}</small>`; b.onclick = () => send(q); $('suggest').appendChild(b);
  }
  $('modelSel').addEventListener('change', () => {
    const k = $('modelSel').value; if (!MODELS[k] || k === modelKey) return;
    if (busy) { $('modelSel').value = modelKey; return; }
    modelKey = k; try { localStorage.setItem('nq-chat-model', k); } catch (e) {}
    $('msgs').innerHTML = ''; newConv(); startWorker();
  });
  loadConvs(); newConv(); renderList(); startWorker();
})();
