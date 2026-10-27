'use strict';
// N-Q-26M-JEV-1027 demo page: builds {name: {kind, question, options}} from the form and asks the worker to decide.
(function () {
  const V = 'nq26m-10', MODEL_V = 'jev-1027';   // code version / weights version (bumping MODEL_V forces a re-download)
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const PRESETS = [
    { name: 'Обращение в поддержку', state: 'Заказ пришёл с разбитым экраном, а поддержка третий день не отвечает. Верните деньги!',
      qs: [['choice', 'Какая тональность у текста?', 'нейтральная, положительная, отрицательная'], ['noul', 'Есть ли в сообщении токсичность?', 'нет, да'],
           ['choice', 'О чём этот запрос?', 'finance, coding, travel, health, cooking, career']] },
    { name: 'Ответ по делу?', state: 'Как сварить борщ?\n---\nВозьмите свёклу, капусту, картофель и мясной бульон; варите около полутора часов.',
      qs: [['noul', 'Отвечает ли ответ на вопрос пользователя?', 'нет, да'], ['score', 'Оцени качество ответа ассистента.', 'плохо, нормально, отлично']] },
    { name: 'Not an answer', state: 'How do I reverse a list in Python?\n---\nThe Eiffel Tower is located in Paris and was built in 1889.',
      qs: [['noul', "Does the reply answer the user's question?", 'no, yes']] },
    { name: 'News topic', state: 'The central bank raised interest rates by half a point on Thursday, citing persistent inflation.',
      qs: [['choice', 'What is this news article about?', 'world, sports, business, science and technology']] },
    { name: 'Review', state: 'Absolutely loved it — the battery lasts two days and the screen is gorgeous.',
      qs: [['choice', 'What is the sentiment of this text?', 'negative, positive']] },
    { name: 'Сложность', state: 'Докажи, что существует бесконечно много простых чисел, и оцени плотность простых до N.',
      qs: [['score', 'Насколько сложен этот запрос?', 'легко, средне, сложно'], ['choice', 'О чём этот запрос? Выбери тему.', 'math, coding, writing, cooking, travel, languages']] },
    { name: 'NLI', state: 'Посылка: Все кошки в доме спали на диване.\nГипотеза: В доме нет ни одной кошки.',
      qs: [['choice', 'Как гипотеза соотносится с посылкой?', 'следует, нейтрально, противоречит']] },
    { name: 'Нужен калькулятор? (новое)', state: 'Сколько будет 347 умножить на 86, а потом разделить на 4?',
      qs: [['noul', 'Понадобится ли для ответа точное вычисление калькулятором?', 'нет, да'], ['noul', 'Нужно ли рассуждать перед ответом на этот запрос?', 'нет, да']] },
    { name: 'Просто поболтать (новое)', state: 'Привет! Как настроение?',
      qs: [['noul', 'Понадобится ли для ответа точное вычисление калькулятором?', 'нет, да'], ['noul', 'Нужно ли рассуждать перед ответом на этот запрос?', 'нет, да']] },
    { name: 'Fact check (new)', state: 'Text: The Eiffel Tower was completed in 1889 and is 330 metres tall.\nClaim: The Eiffel Tower was completed in 1925.',
      qs: [['noul', 'Does the text support the claim?', 'no, yes']] },
    { name: 'Tricky NLI (new)', state: 'Premise: A man is playing a guitar on stage in front of a crowd.\nHypothesis: Nobody is making music.',
      qs: [['choice', 'How does the hypothesis relate to the premise?', 'entailment, neutral, contradiction']] },
  ];

  function addRow(kind = 'choice', q = '', opts = '') {
    const row = document.createElement('div'); row.className = 'qrow';
    row.innerHTML = `<select><option value="choice">choice</option><option value="score">score</option><option value="noul">yes / no</option></select>
      <input class="q" placeholder="question" /><input class="o" placeholder="option 1, option 2, …" /><button class="del" type="button" aria-label="remove">×</button>`;
    row.querySelector('select').value = kind; row.querySelector('.q').value = q; row.querySelector('.o').value = opts;
    const sync = () => { const noul = row.querySelector('select').value === 'noul'; const o = row.querySelector('.o'); if (noul && !o.value.trim()) o.value = 'no, yes'; o.title = noul ? 'two labels: "no" first, "yes" second' : ''; };
    row.querySelector('select').addEventListener('change', sync); sync();
    row.querySelector('.del').onclick = () => row.remove();
    $('questions').appendChild(row);
  }
  function loadPreset(p, btn) {
    $('state').value = p.state; $('questions').innerHTML = ''; p.qs.forEach((q) => addRow(...q)); $('results').innerHTML = '';
    document.querySelectorAll('.presets button').forEach((b) => b.classList.toggle('is-on', b === btn));
  }
  PRESETS.forEach((p, i) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = p.name; b.onclick = () => loadPreset(p, b); $('presets').appendChild(b); if (!i) loadPreset(p, b); });
  $('addQ').onclick = () => addRow();

  const worker = new Worker(`../nq-runtime/nq_worker.js?v=${V}`);
  let ready = false, pending = null;
  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'progress') $('status').textContent = m.cached ? 'loading from browser cache…' : `downloading model.qpack ${(m.got / 1e6).toFixed(1)}${m.total ? ' / ' + (m.total / 1e6).toFixed(1) : ''} MB…`;
    else if (m.type === 'status' && !ready) $('status').textContent = 'unpacking, self-test…';
    else if (m.type === 'status' && pending) $('timing').textContent = `… ${m.text}`;
    else if (m.type === 'ready') {
      ready = true; $('run').disabled = false; $('status').textContent = `ready · runs in this tab (${m.format}, ${({ 'wasm-core': 'WebAssembly SIMD core', wasm: 'WebAssembly SIMD', js: 'plain JavaScript' })[m.backend]}, Web Worker)`;
      $('selftest').textContent = `self-test ${m.selftest[0]}/${m.selftest[1]} ${m.selftest[0] === m.selftest[1] ? '✓' : '✗'}${m.decision ? '' : ' · ✗ no decision head in this model'}`;
    } else if (m.type === 'decision' && pending) { render(pending, m.result); $('timing').textContent = `${(m.ms / 1000).toFixed(1)} s`; pending = null; $('run').disabled = false; }
    else if (m.type === 'error') { $('status').textContent = 'error: ' + m.message; pending = null; $('run').disabled = !ready; }
  };
  worker.postMessage({ type: 'load', url: new URL(`model.q8.qpack?v=${MODEL_V}`, location.href).href, version: MODEL_V });   // absolute: the worker resolves URLs from /nq-runtime/

  $('run').onclick = () => {
    const qs = {}, meta = [];
    document.querySelectorAll('.qrow').forEach((row, i) => {
      const kind = row.querySelector('select').value, question = row.querySelector('.q').value.trim();
      const options = row.querySelector('.o').value.split(',').map((s) => s.trim()).filter(Boolean);
      if (!question || (kind !== 'noul' && options.length < 2)) return;
      const name = 'q' + i; qs[name] = kind === 'score' ? { kind, question, tiers: options } : { kind, question, options: kind === 'noul' ? (options.length === 2 ? options : ['no', 'yes']) : options };
      meta.push({ name, kind, question, options: qs[name].options || qs[name].tiers });
    });
    if (!meta.length || !$('state').value.trim()) return;
    pending = meta; $('run').disabled = true; $('results').innerHTML = ''; $('timing').textContent = '…';
    worker.postMessage({ type: 'decide', state: $('state').value, questions: qs, symmetric: $('sym').checked });
  };

  function render(meta, result) {
    $('results').innerHTML = meta.map(({ name, kind, question, options }) => {
      const r = result[name];
      const probs = kind === 'noul' ? { [options[0]]: 1 - r.p_yes, [options[1]]: r.p_yes } : r.probs;
      const top = Object.entries(probs).sort((a, b) => b[1] - a[1])[0];
      const bars = Object.entries(probs).map(([o, p]) => `<div class="pbar${o === top[0] ? ' is-top' : ''}"><span class="pbar__name" title="${esc(o)}">${esc(o)}</span>
        <span class="pbar__track"><span class="pbar__fill" style="display:block;width:${(100 * p).toFixed(1)}%"></span></span><span class="pbar__p">${(100 * p).toFixed(1)}%</span></div>`).join('');
      return `<div class="res"><div class="res__q">${esc(kind)} · ${esc(question)}</div><div class="res__a">${esc(top[0])}<small>${(100 * top[1]).toFixed(1)}%</small></div>${bars}</div>`;
    }).join('');
  }
})();
