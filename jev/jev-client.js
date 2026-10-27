'use strict';
// Shared JEV client for the /jev/* use-case pages: one Web Worker running N-Q-26M-JEV-1027 (QPACK2, WASM SIMD core),
// a promise queue over {type:'decide'} messages, a status pill and small helpers (language, escaping).
(function () {
  const SRC = document.currentScript.src;
  const V = 'nq26m-10', MODEL_V = 'jev-1027';
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const lang = () => { const l = (document.documentElement.getAttribute('lang') || 'en').slice(0, 2); return ['ru', 'en', 'zh'].includes(l) ? l : 'en'; };
  // trilingual inline text: every language is rendered, main.js shows the active one
  const L = (ru, en, zh) => ['ru', 'en', 'zh'].map((l, i) => `<span data-lang="${l}"${lang() === l ? ' class="is-active"' : ''}>${esc([ru, en, zh || en][i])}</span>`).join('');
  const isRu = (t) => (String(t).match(/[а-яё]/gi) || []).length > 3;
  const pct = (p) => `${Math.round(100 * p)}%`;

  class JevClient {
    constructor(statusEl) {
      this.queue = []; this.busy = null; this.ready = false; this.statusEl = statusEl; this.readyCbs = [];
      this.w = new Worker(new URL(`../nq-runtime/nq_worker.js?v=${V}`, SRC).href);
      this.w.onmessage = (ev) => this.onMsg(ev.data);
      this.status('loading', L('загрузка модели…', 'loading the model…', '正在加载模型…'));
      this.w.postMessage({ type: 'load', url: new URL(`model.q8.qpack?v=${MODEL_V}`, SRC).href, version: MODEL_V });
    }
    status(kind, html) { if (!this.statusEl) return; this.statusEl.className = `jev-pill is-${kind}`; this.statusEl.innerHTML = `<i></i>${html}`; }
    onMsg(m) {
      if (m.type === 'progress') this.status('loading', m.cached ? L('загрузка из кэша браузера…', 'loading from the browser cache…', '从浏览器缓存加载…')
        : `${L('скачиваю модель', 'downloading the model', '正在下载模型')} ${(m.got / 1e6).toFixed(1)}${m.total ? ' / ' + (m.total / 1e6).toFixed(1) : ''} MB`);
      else if (m.type === 'ready') {
        this.ready = true;
        this.status('ready', `N-Q-26M-JEV-1027 · ${L('в этой вкладке', 'in this tab', '在此标签页中')} · self-test ${m.selftest[0]}/${m.selftest[1]}`);
        this.readyCbs.forEach((f) => f()); this.readyCbs = [];
      } else if (m.type === 'decision' && this.busy) { const b = this.busy; this.busy = null; b.resolve({ ...m.result, _ms: m.ms }); this.next(); }
      else if (m.type === 'error') {
        if (this.busy) { const b = this.busy; this.busy = null; b.reject(new Error(m.message)); this.next(); }
        else this.status('error', 'error: ' + esc(m.message));
      }
    }
    onReady(f) { if (this.ready) f(); else this.readyCbs.push(f); }
    next() {
      if (this.busy || !this.queue.length) return;
      this.busy = this.queue.shift();
      this.w.postMessage({ type: 'decide', state: this.busy.state, questions: this.busy.questions, symmetric: this.busy.symmetric });
    }
    decide(state, questions, symmetric = true) {
      return new Promise((resolve, reject) => { this.queue.push({ state, questions, symmetric, resolve, reject }); this.next(); });
    }
  }

  // the exact phrasings JEV was trained with (nq/dataset/prep_jev*.py) — other wordings work, these work best
  const Q = {
    topic: { ru: 'О чём этот запрос? Выбери тему.', en: 'What is this request about? Pick the topic.' },
    complexity: { ru: 'Насколько сложен этот запрос?', en: 'How difficult is this request?' },
    tiers: { ru: ['легко', 'средне', 'сложно'], en: ['easy', 'medium', 'hard'] },
    calc: { ru: 'Понадобится ли для ответа точное вычисление калькулятором?', en: 'Does answering this need an exact calculator computation?' },
    think: { ru: 'Нужно ли рассуждать перед ответом на этот запрос?', en: 'Should the assistant reason step by step before answering?' },
    toxic: { ru: 'Есть ли в сообщении токсичность?' },
    sentiment: { ru: 'Какая тональность у текста?', en: 'What is the sentiment of this text?' },
    sentOpts: { ru: ['нейтральная', 'положительная', 'отрицательная'], en: ['negative', 'positive'] },
    match: { ru: 'Отвечает ли ответ на вопрос пользователя?', en: "Does the reply answer the user's question?" },
    reply: { ru: 'Какой из ответов отвечает на этот вопрос?', en: 'Which reply answers this question?' },
    quality: { ru: 'Оцени качество ответа ассистента.', en: "Rate the quality of the assistant's answer." },
    qTiers: { ru: ['плохо', 'нормально', 'отлично'], en: ['bad', 'okay', 'excellent'] },
    pref: { ru: 'Какой ответ на запрос лучше?', en: 'Which answer to the request is better?' },
    claim: 'Does the text support the claim?',
    nli: { ru: 'Как гипотеза соотносится с посылкой?', en: 'How does the hypothesis relate to the premise?' },
    nliOpts: { ru: ['следует', 'нейтрально', 'противоречит'], en: ['entailment', 'neutral', 'contradiction'] },
    yn: { ru: ['нет', 'да'], en: ['no', 'yes'] },
  };
  const TOPICS = ['chit_chat', 'coding', 'math', 'stem', 'writing', 'facts', 'health', 'cooking', 'travel', 'finance', 'career', 'law', 'languages', 'entertainment'];
  const TOPIC_NAMES = { chit_chat: ['болтовня', 'chit-chat', '闲聊'], coding: ['программирование', 'coding', '编程'], math: ['математика', 'math', '数学'],
    stem: ['наука', 'science', '科学'], writing: ['тексты', 'writing', '写作'], facts: ['факты', 'facts', '事实'], health: ['здоровье', 'health', '健康'],
    cooking: ['кулинария', 'cooking', '烹饪'], travel: ['путешествия', 'travel', '旅行'], finance: ['финансы', 'finance', '金融'], career: ['карьера', 'career', '职业'],
    law: ['право', 'law', '法律'], languages: ['языки', 'languages', '语言'], entertainment: ['развлечения', 'entertainment', '娱乐'] };

  window.Jev = { JevClient, esc, lang, L, isRu, pct, Q, TOPICS, TOPIC_NAMES, cut: (t, n) => { t = String(t).replace(/\s+/g, ' ').trim(); return t.length <= n ? t : t.slice(0, n).replace(/\s\S*$/, '') + ' …'; } };
})();
