'use strict';
// Experimental chat: N-Q-26M-Chat + the JEV decision model (second worker). JEV tags each request (topic, difficulty,
// toxicity), checks whether each answer addresses the question and re-samples off-topic answers — all shown to the user
// and switchable in settings. Otherwise identical to /chat/ (chat.js).
// N-Q-26M-Chat page: UI only. The model runs in ../nq-runtime/nq_worker.js (Web Worker); conversations live in localStorage.
// A conversation: {id, title, updated, messages: [{role:'user', content} | {role:'assistant', raw, segs:[{kind, text}], tools:[...], meta}]}
// `raw` is the assistant turn with its control tokens (<THINK>, <CALC>…) — that is what the model re-reads.
(function () {
  const V = 'nq26m-8', MODEL_V = 'nq26m-q8-1';   // code version / weights version (bumping MODEL_V forces a re-download)
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
  const KEY = 'nq-chatexp-convs';
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
    if (m.jevCheck === 'pending') parts.push(`<div><span class="jev-check is-pending"><b>JEV</b>${L('проверяет ответ…', 'checking the answer…', '正在检查回答…')}</span></div>`);
    else if (typeof m.jevCheck === 'number') parts.push(`<div><span class="jev-check ${m.jevCheck >= 0.5 ? 'is-good' : 'is-bad'}" title="P(answer addresses the question)"><b>JEV</b>${m.jevCheck >= 0.5 ? L('по делу', 'on topic', '切题') : L('не по делу', 'off topic', '不切题')} ${(100 * m.jevCheck).toFixed(0)}%</span></div>`);
    if (m.jevNote) parts.push(`<div class="m__note">${m.jevNote}</div>`);
    return parts.join('') || `<div class="md"></div>`;
  }
  function msgEl(m, idx) {
    const d = document.createElement('div');
    if (m.role === 'user') {
      d.className = 'm m--user'; d.style.flexDirection = 'column'; d.style.alignItems = 'flex-end';
      d.innerHTML = '<div class="m__body"></div>'; d.firstChild.textContent = m.content;
      if (m.jev) {
        const t = document.createElement('div'); t.className = 'jev-tags';
        // label: trilingual html from L(); value and confidence from JEV
        const tag = (label, v, p) => `<span class="jev-tag"${p != null ? ` title="JEV confidence ${(100 * p).toFixed(0)}%"` : ''}><b>JEV</b><span>${label}: ${esc(v)}</span></span>`;
        t.innerHTML = (m.jev.topic ? tag(L('тема', 'topic', '主题'), m.jev.topic.choice, m.jev.topic.confidence) : '') +
          (m.jev.level ? tag(L('сложность', 'difficulty', '难度'), m.jev.level.level, m.jev.level.confidence) : '') +
          (m.jev.toxic ? tag(L('токсичность', 'toxicity', '毒性'), `${(100 * m.jev.toxic.p_yes).toFixed(0)}%`, null) : '');
        d.appendChild(t);
      }
      return d;
    }
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

  // ---------------------------------------------------------------- JEV (second worker)
  const jevWorker = new Worker(`../nq-runtime/nq_worker.js?v=${V}`);
  let jevReady = false; const jevQueue = []; let jevBusy = null;
  const jevOn = () => $('jevOn').checked && jevReady;
  jevWorker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'ready') { jevReady = true; jevPill(); }
    else if (m.type === 'decision' && jevBusy) { const r = jevBusy; jevBusy = null; r.resolve(m.result); jevNext(); }
    else if (m.type === 'error') { if (jevBusy) { const r = jevBusy; jevBusy = null; r.resolve(null); jevNext(); } else jevPill('error'); }
  };
  jevWorker.postMessage({ type: 'load', url: new URL(`../jev/model.q8.qpack?v=${MODEL_V}`, location.href).href, version: MODEL_V });
  function jevNext() { if (jevBusy || !jevQueue.length) return; jevBusy = jevQueue.shift(); jevWorker.postMessage({ type: 'decide', state: jevBusy.state, questions: jevBusy.questions }); }
  const jevDecide = (state, questions) => new Promise((resolve) => { jevQueue.push({ state, questions, resolve }); jevNext(); });
  const isRu = (t) => (t.match(/[а-яё]/gi) || []).length > 3;
  const cut = (t, n) => { t = t.replace(/\s+/g, ' ').trim(); return t.length <= n ? t : t.slice(0, n).replace(/\s\S*$/, '') + ' …'; };
  const TOPICS = ['chit_chat', 'coding', 'math', 'stem', 'writing', 'facts', 'health', 'cooking'];   // labels from JEV's training topics
  const RETRY_BELOW = +(new URLSearchParams(location.search).get('jev_retry_below') || 0.35);   // URL override for testing
  // the exact question phrasings and option labels JEV was trained with (prep_jev.py)
  function jevTagsFor(text) {
    const ru = isRu(text), q = {
      topic: { kind: 'choice', question: ru ? 'О чём этот запрос? Выбери тему.' : 'What is this request about? Pick the topic.', options: TOPICS },
      level: { kind: 'score', question: ru ? 'Насколько сложен этот запрос?' : 'How difficult is this request?', tiers: ru ? ['легко', 'средне', 'сложно'] : ['easy', 'medium', 'hard'] } };
    if (ru) q.toxic = { kind: 'noul', question: 'Есть ли в сообщении токсичность?', options: ['нет', 'да'] };
    return jevDecide(cut(text, 700), q);
  }
  function jevMatch(question, answer) {
    const ru = isRu(question);
    return jevDecide(`${cut(question, 500)}\n---\n${cut(answer, 300)}`, { m: { kind: 'noul', question: ru ? 'Отвечает ли ответ на вопрос пользователя?' : "Does the reply answer the user's question?", options: ru ? ['нет', 'да'] : ['no', 'yes'] } })
      .then((r) => (r ? r.m.p_yes : null));
  }
  function jevPill(state) {
    const p = $('jevPill'); const on = $('jevOn').checked;
    p.classList.toggle('is-off', !on); p.classList.toggle('is-ready', on && jevReady); p.classList.toggle('is-busy', on && !jevReady);
    $('jevText').textContent = state === 'error' ? 'JEV: error' : !on ? 'JEV: off' : jevReady ? 'JEV: on' : 'JEV: loading';
    $('jevSub').classList.toggle('is-off', !on);
  }
  const visibleText = (m) => (m.segs || []).filter((x) => x.kind === 'text').map((x) => x.text).join('\n').trim();

  // ---------------------------------------------------------------- worker
  const worker = new Worker(`../nq-runtime/nq_worker.js?v=${V}`);
  let ready = false, busy = false, live = null, lastAnswered = null, lastMemory = true;
  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'progress') {
      if (m.cached) { $('loadText').textContent = 'Loading from browser cache…'; $('loadBar').style.width = '100%'; }
      else { const pct = m.total ? Math.round(100 * m.got / m.total) : 0; $('loadBar').style.width = pct + '%'; $('loadText').textContent = `Downloading the model… ${(m.got / 1e6).toFixed(1)}${m.total ? ' / ' + (m.total / 1e6).toFixed(1) : ''} MB`; }
    } else if (m.type === 'status' && !ready) { $('loadText').textContent = 'Unpacking and self-testing…'; }
    else if (m.type === 'ready') {
      ready = true; $('loading').hidden = true; setStatus('ready');
      $('modelMeta').innerHTML = `${m.layers} layers · hidden ${m.hidden} · vocab ${m.vocab.toLocaleString()}<br>self-test ${m.selftest[0]}/${m.selftest[1]} ${m.selftest[0] === m.selftest[1] ? '✓' : '✗'} · ${m.format} · ${m.backend === 'wasm' ? 'WASM SIMD' : 'JS'}`;
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
      afterAnswer();
    } else if (m.type === 'error') {
      if (!ready) { $('loadText').textContent = 'Error: ' + m.message; }
      else if (live) { live.note = 'error: ' + esc(m.message); finishLive(); }
      setStatus(ready ? 'ready' : 'error');
    }
  };
  worker.postMessage({ type: 'load', url: new URL(`../chat/model.q8.qpack?v=${MODEL_V}`, location.href).href, version: MODEL_V });   // absolute: the worker resolves URLs from /nq-runtime/

  function setStatus(s) {
    const p = $('statusPill'); p.classList.toggle('is-ready', s === 'ready'); p.classList.toggle('is-busy', s === 'busy');
    $('statusText').textContent = { ready: 'ready', busy: 'generating', loading: 'loading', error: 'error' }[s] || s;
  }
  function redrawLive() {
    const el = $('msgs').lastElementChild; if (!el) return;
    el.querySelector('.m__content').innerHTML = botBody(live); scrollDown(false);
  }
  // JEV after each answer: score it; if off topic and retries are on, sample again (same turn, worker snapshot) and keep the best
  let candidates = [], turnQuestion = '';
  async function afterAnswer() {
    const ok = jevOn() && $('jevCheck').checked && visibleText(live);
    if (!ok) { finishLive(); setStatus('ready'); return; }
    live.jevCheck = 'pending'; redrawLive(); setStatus('busy');
    const p = await jevMatch(turnQuestion, visibleText(live));
    live.jevCheck = p == null ? undefined : p; candidates.push(live);
    const canRetry = $('jevRetry').checked && !opts().greedy && p != null && p < RETRY_BELOW && candidates.length < 3;
    if (canRetry) {
      live = { role: 'assistant', segs: [], raw: null };
      cur.messages[cur.messages.length - 1] = live; redrawLive();
      worker.postMessage({ type: 'regenerate', opts: opts() }); return;
    }
    if (candidates.length > 1) {
      const best = candidates.reduce((a, b) => ((b.jevCheck ?? -1) > (a.jevCheck ?? -1) ? b : a));
      const rejected = candidates.filter((c) => c !== best).map((c) => `${(100 * c.jevCheck).toFixed(0)}%`).join(', ');
      best.jevNote = L(`JEV отклонил ${candidates.length - 1} вариант(а) с оценкой «по делу» ${rejected} и показал лучший`, `JEV rejected ${candidates.length - 1} candidate(s) rated ${rejected} on topic and kept the best`, `JEV 拒绝了 ${candidates.length - 1} 个切题度为 ${rejected} 的候选，保留了最佳回答`);
      live = best; cur.messages[cur.messages.length - 1] = best;
    }
    candidates = []; finishLive(); setStatus('ready');
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
    const um = { role: 'user', content: text }; cur.messages.push(um); $('empty').hidden = true;
    $('msgs').appendChild(msgEl(um)); touch();
    turnQuestion = text; candidates = [];
    if (jevOn() && $('jevTags').checked) jevTagsFor(text).then((r) => { if (r) { um.jev = r; saveConvs(); if (!busy) renderAll(); else { const els = $('msgs').querySelectorAll('.m--user'); const el = els[els.length - 1]; if (el && cur.messages.includes(um)) el.replaceWith(msgEl(um)); } } });
    startLive(); worker.postMessage({ type: 'chat', turns: prior, text, opts: opts() });
  }
  function regenerate() {
    if (busy || !cur.messages.length || cur.messages[cur.messages.length - 1].role !== 'assistant') return;
    const prior = turnsOf(cur.messages.slice(0, -2)), text = cur.messages[cur.messages.length - 2].content; turnQuestion = text; candidates = [];
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
  ['jevOn', 'jevCheck', 'jevRetry', 'jevTags'].forEach((id) => $(id).addEventListener('change', () => { jevPill(); try { localStorage.setItem('nq-chatexp-jev', JSON.stringify(['jevOn', 'jevCheck', 'jevRetry', 'jevTags'].map((k) => $(k).checked))); } catch (e) {} }));
  try { const j = JSON.parse(localStorage.getItem('nq-chatexp-jev') || 'null'); if (j) ['jevOn', 'jevCheck', 'jevRetry', 'jevTags'].forEach((k, i) => { $(k).checked = j[i]; }); } catch (e) {}
  $('jevPill').addEventListener('click', (e) => { e.stopPropagation(); $('jevOn').checked = !$('jevOn').checked; $('jevOn').dispatchEvent(new Event('change')); });
  try { if (localStorage.getItem('nq-chatexp-banner') === 'closed') $('jevBanner').hidden = true; } catch (e) {}
  $('bannerClose').addEventListener('click', () => { $('jevBanner').hidden = true; try { localStorage.setItem('nq-chatexp-banner', 'closed'); } catch (e) {} });
  jevPill();
  function savePrefs() { try { localStorage.setItem('nq-chatexp-prefs', JSON.stringify(opts())); } catch (e) {} }
  try {
    const p = JSON.parse(localStorage.getItem('nq-chatexp-prefs') || 'null');
    if (p) { $('temp').value = p.temperature; $('topP').value = p.top_p; $('maxNew').value = p.max_new; $('rep').value = p.rep; $('greedy').checked = p.greedy; $('think').checked = p.think; $('memory').checked = p.memory;
      for (const [id, out, d] of [['temp', 'tempVal', 2], ['topP', 'topPVal', 2], ['maxNew', 'maxNewVal', 0], ['rep', 'repVal', 2]]) $(out).textContent = (+$(id).value).toFixed(d); }
  } catch (e) {}
  for (const [q, ru, en] of SUGGEST) {
    const b = document.createElement('button'); b.type = 'button'; b.innerHTML = `${esc(q)}<small>${L(ru, en, en)}</small>`; b.onclick = () => send(q); $('suggest').appendChild(b);
  }
  loadConvs(); newConv(); renderList();
})();
