// N-Q-JEV decisions in the browser: a port of NQ4ForDecision.encode / decide (nq/model/modeling_nq4.py).
//   <bos><USER>state<EOT><ASK>question<OPT>option_1<OPT>…option_K<OPT>  -> score_k = head(hidden at the <OPT> closing option k)
// The runtime is recurrent, so shared prefixes are computed once and branched with snapshot()/restore():
// the state for all questions, and state+question for both option orders of the symmetric mode.
'use strict';

(function () {
  const Lib = (typeof window !== 'undefined' ? (window.QOmniLib = window.QOmniLib || {}) : (globalThis.QOmniLib = globalThis.QOmniLib || {}));
  const OPT = 8203, ASK = 8204, BOS = 1, USER = 8192, EOT = 8194;
  const logSoftmax = (xs) => { const m = Math.max(...xs); const l = Math.log(xs.reduce((s, x) => s + Math.exp(x - m), 0)) + m; return xs.map((x) => x - l); };

  // options in the given order -> log-probabilities (same order); the model is positioned right after the question's <OPT>
  function scoreOptions(model, tok, options, tick) {
    const scores = [];
    for (const o of options) {
      for (const id of tok.encode(o).slice(0, 160)) { model.step(id, false); tick(); }
      scores.push(model.decisionScore(model.step(OPT, 'hidden'))); tick();
    }
    return logSoftmax(scores);
  }

  // questions: {name: {kind: 'choice'|'score'|'noul', question, options | tiers}} -> {name: result} (same shape as Python decide)
  function decide(model, tok, state, questions, { symmetric = true, onProgress = null } = {}) {
    let done = 0; const tick = () => { done++; if (onProgress && done % 8 === 0) onProgress(done); };
    model.reset();
    for (const id of [BOS, USER, ...tok.encode(state).slice(0, 640), EOT]) { model.step(id, false); tick(); }
    const afterState = model.snapshot(), out = {};
    for (const [name, q] of Object.entries(questions)) {
      const opts = (q.options || q.tiers || (q.kind === 'noul' ? ['no', 'yes'] : [])).slice();
      model.restore(afterState);
      for (const id of [ASK, ...tok.encode(q.question).slice(0, 96), OPT]) { model.step(id, false); tick(); }
      const afterQ = model.snapshot();
      let lp = scoreOptions(model, tok, opts, tick);
      if (symmetric && opts.length > 1) {
        model.restore(afterQ);
        const rev = scoreOptions(model, tok, opts.slice().reverse(), tick).reverse();
        lp = lp.map((x, i) => (x + rev[i]) / 2);
      }
      const p = logSoftmax(lp).map(Math.exp); let k = 0; p.forEach((x, i) => { if (x > p[k]) k = i; });
      if (q.kind === 'noul') out[name] = { p_yes: p[p.length - 1] };
      else out[name] = { [q.kind === 'choice' ? 'choice' : 'level']: opts[k], index: k, probs: Object.fromEntries(opts.map((o, i) => [o, p[i]])), confidence: p[k] };
    }
    return out;
  }

  Lib.decide = decide;
  if (typeof module !== 'undefined') module.exports = { decide };
})();
