// Exact circuits for N-Q in the browser: a port of nq/model/circuits.py (evaluate + CircuitExecutor) with Python
// semantics where they matter: exact integers (BigInt), floor division/modulo, round-half-even, str(float) formatting.
// The model writes <CALC>expr<EQ>; the executor decodes expr with the tokenizer, evaluates it and forces
// tokenizer.encode(result) + <ECALC>. Checked against the Python evaluate() on corpus spans (web_tests/test_circuits.cjs).
'use strict';

(function () {
  const Lib = (typeof window !== 'undefined' ? (window.QOmniLib = window.QOmniLib || {}) : (globalThis.QOmniLib = globalThis.QOmniLib || {}));
  const TEXT_VOCAB = 8192;
  const [USER, MODEL, EOT, THINK, ETHINK, CALC, EQ, ECALC] = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => TEXT_VOCAB + i);
  const BOS = 1, EOS = 2;
  const UNITS = { 'km>mi': 0.621371, 'mi>km': 1.609344, 'kg>lb': 2.204623, 'lb>kg': 0.453592, 'm>ft': 3.28084, 'ft>m': 0.3048,
    'l>gal': 0.264172, 'gal>l': 3.78541, 'cm>in': 0.393701, 'in>cm': 2.54 };
  const WD_EN = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  const WD_RU = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье'];

  // ---- Python number model: {int: BigInt} or {float: number}
  const I = (v) => ({ t: 'i', v: BigInt(v) }), F = (v) => ({ t: 'f', v: Number(v) });
  const num = (x) => (x.t === 'i' ? Number(x.v) : x.v);

  function pyFloat(s) {                 // float(s): surrounding whitespace allowed, otherwise strict
    const t = String(s).trim();
    if (!/^[+-]?(\d+\.?\d*(e[+-]?\d+)?|\.\d+(e[+-]?\d+)?|inf(inity)?|nan)$/i.test(t)) throw new Error('float: ' + s);
    return Number(/inf/i.test(t) ? (t[0] === '-' ? -Infinity : Infinity) : /nan/i.test(t) ? NaN : t);
  }
  function pyInt(s) { const t = String(s).trim(); if (!/^[+-]?\d+$/.test(t)) throw new Error('int: ' + s); return BigInt(t); }

  function pyRound(x, n) {              // Python round(float, n): correctly rounded, exact ties go to the even digit
    if (!isFinite(x) || Math.abs(x) >= 1e21) return x;
    const exact = Math.abs(x).toFixed(Math.min(100, n + 70));       // exact decimal expansion of the double (enough digits here)
    const dot = exact.indexOf('.'), tail = exact.slice(dot + 1 + n);
    if (/^50*$/.test(tail)) {                                         // a true tie: keep the truncated value if its last digit is even
      const trunc = exact.slice(0, dot + 1 + n), last = Number(trunc[trunc.length - 1]);
      if (last % 2 === 0) return Math.sign(x) * Number(trunc);
    }
    return Number(x.toFixed(n));                                      // toFixed is correctly rounded off-tie (ties: away from zero)
  }

  function pyStrFloat(x) {              // Python repr/str of a float (shortest round-trip)
    if (Number.isNaN(x)) return 'nan';
    if (!isFinite(x)) return x > 0 ? 'inf' : '-inf';
    if (Number.isInteger(x) && Math.abs(x) < 1e16) return x.toFixed(1);
    let s = String(x);
    if (/e/.test(s)) { let [m, e] = s.split('e'); const ei = Number(e); if (Math.abs(x) >= 1e16 || Math.abs(x) < 1e-4) return `${m}e${ei < 0 ? '-' : '+'}${String(Math.abs(ei)).padStart(2, '0')}`; }
    if (Math.abs(x) < 1e-4 && x !== 0) { const [m, e] = x.toExponential().split('e'); const ei = Number(e); return `${m}e${ei < 0 ? '-' : '+'}${String(Math.abs(ei)).padStart(2, '0')}`; }
    return s;
  }

  function fmtNum(x) {                  // fmt_num: round floats to 6 places, integral values without ".0"
    if (x.t === 'i') return x.v.toString();
    const v = pyRound(x.v, 6);
    if (Number.isInteger(v)) return BigInt(v).toString();
    return pyStrFloat(v);
  }
  const fmtF = (v) => fmtNum(F(v));

  // ---- Python expression subset: numbers, + - * / // % ** (pow exponent |b| <= 12), unary -/+, parentheses
  function parseExpr(src) {
    const toks = []; const D_ = '\\d(?:_?\\d)*';   // Python literals allow single underscores between digits
    const re = new RegExp(`\\s*(${D_}\\.(?:${D_})?(?:[eE][+-]?${D_})?|\\.${D_}(?:[eE][+-]?${D_})?|${D_}(?:[eE][+-]?${D_})?|\\*\\*|\\/\\/|[-+*/%()])`, 'y');
    let m, i = 0;
    while (i < src.length) {
      if (/^\s*$/.test(src.slice(i))) break;
      re.lastIndex = i; m = re.exec(src); if (!m) throw new Error('syntax'); toks.push(m[1].replace(/_/g, '')); i = re.lastIndex;
    }
    let p = 0;
    const peek = () => toks[p], eat = (t) => { if (toks[p] !== t) throw new Error('syntax'); p++; };
    function atom() {
      const t = toks[p++];
      if (t === undefined) throw new Error('syntax');
      if (t === '(') { const v = sum(); eat(')'); return v; }
      if (/^\d/.test(t) || t[0] === '.') {
        if (/^\d+$/.test(t)) { if (t.length > 1 && t[0] === '0' && /[1-9]/.test(t)) throw new Error('leading zero'); return I(t); }
        return F(Number(t));
      }
      throw new Error('syntax');
    }
    function power() { const a = atom(); if (peek() === '**') { p++; const b = unary(); return binop('**', a, b); } return a; }
    function unary() {   // only unary minus: the Python evaluator has no UAdd ("+3" is rejected there too)
      if (peek() === '+') throw new Error('unary plus');
      if (peek() === '-') { p++; const v = unary(); return v.t === 'i' ? I(-v.v) : F(-v.v); }
      return power();
    }
    function term() { let a = unary(); while (['*', '/', '//', '%'].includes(peek())) { const op = toks[p++]; a = binop(op, a, unary()); } return a; }
    function sum() { let a = term(); while (peek() === '+' || peek() === '-') { const op = toks[p++]; a = binop(op, a, term()); } return a; }
    const v = sum(); if (p !== toks.length) throw new Error('syntax'); return v;
  }

  function binop(op, a, b) {
    const ints = a.t === 'i' && b.t === 'i';
    if (op === '**') {
      if (Math.abs(num(b)) > 12) throw new Error('pow too large');
      if (ints && b.v >= 0n) return I(a.v ** b.v);
      if (ints && a.v === 0n) throw new Error('zero division');
      return F(Math.pow(num(a), num(b)));
    }
    if (op === '/') { if (num(b) === 0) throw new Error('zero division'); return F(num(a) / num(b)); }
    if (op === '//' || op === '%') {
      if (num(b) === 0) throw new Error('zero division');
      if (ints) { let q = a.v / b.v; if ((a.v % b.v !== 0n) && ((a.v < 0n) !== (b.v < 0n))) q -= 1n; return op === '//' ? I(q) : I(a.v - q * b.v); }
      const x = num(a), y = num(b), q = Math.floor(x / y); return op === '//' ? F(q) : F(x - q * y);
    }
    if (ints) return I(op === '+' ? a.v + b.v : op === '-' ? a.v - b.v : a.v * b.v);
    const x = num(a), y = num(b); return F(op === '+' ? x + y : op === '-' ? x - y : x * y);
  }

  function isoDate(s) { if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) throw new Error('date'); const [y, mo, d] = s.split('-').map(Number);
    const t = Date.UTC(y, mo - 1, d), dt = new Date(t); if (dt.getUTCFullYear() !== y || dt.getUTCMonth() !== mo - 1 || dt.getUTCDate() !== d) throw new Error('date'); return t; }
  const DAY = 86400000;
  const iso = (t) => new Date(t).toISOString().slice(0, 10);
  const weekday = (t) => (new Date(t).getUTCDay() + 6) % 7;   // Monday = 0

  function evaluate(expr) {
    expr = expr.trim();
    try {
      if (expr.startsWith('date:')) { const [base, n] = expr.slice(5).split('+'); if (n === undefined) throw 0; return iso(isoDate(base) + Number(pyInt(n)) * DAY); }
      if (expr.startsWith('wd:')) return WD_EN[weekday(isoDate(expr.slice(3)))];
      if (expr.startsWith('wdru:')) return WD_RU[weekday(isoDate(expr.slice(5)))];
      if (expr.startsWith('dur:')) { const [a, b] = split2(expr.slice(4), ','); return String(Math.abs(Math.round((isoDate(b) - isoDate(a)) / DAY))); }
      if (expr.startsWith('conv:')) {
        const r = expr.slice(5), sp = r.indexOf(' '); if (sp < 0) throw 0;
        const v = pyFloat(r.slice(0, sp)), [a, b] = split2(r.slice(sp + 1), '>');
        if (UNITS[`${a}>${b}`] !== undefined) return fmtF(pyRound(v * UNITS[`${a}>${b}`], 2));
        if (a === 'c' && b === 'f') return fmtF(pyRound(v * 9 / 5 + 32, 2));
        if (a === 'f' && b === 'c') return fmtF(pyRound((v - 32) * 5 / 9, 2));
        return null;
      }
      if (expr.startsWith('cmp:')) { const [a, b] = split2(expr.slice(4), ',').map(pyFloat); return a > b ? 'first' : a < b ? 'second' : 'equal'; }
      if (expr.startsWith('sort:')) { const xs = expr.slice(5).split(','); const keyed = xs.map((x, i) => [pyFloat(x), i, x]); keyed.sort((p, q) => p[0] - q[0] || p[1] - q[1]); return keyed.map((k) => k[2]).join(','); }
      if (expr.startsWith('filter:')) {
        const r = expr.slice(7), c = r.indexOf(':'); if (c < 0) throw 0; const cond = r.slice(0, c), rest = r.slice(c + 1);
        for (const [sym, fn] of [['>=', (x, y) => x >= y], ['<=', (x, y) => x <= y], ['==', (x, y) => x === y], ['>', (x, y) => x > y], ['<', (x, y) => x < y]]) {
          if (cond.startsWith(sym)) { const thr = pyFloat(cond.slice(sym.length)); const xs = rest.split(',').filter((x) => fn(pyFloat(x), thr)); return xs.length ? xs.join(',') : 'none'; }
        }
        return null;
      }
      if (expr.startsWith('count:')) return String([...expr.slice(6)].length);
      for (const [k, fn] of [['mean:', (v) => v.reduce((a, b) => a + b, 0) / v.length], ['max:', (v) => Math.max(...v)], ['min:', (v) => Math.min(...v)], ['sum:', (v) => v.reduce((a, b) => a + b, 0)]]) {
        if (expr.startsWith(k)) return fmtF(pyRound(fn(expr.slice(k.length).split(',').map(pyFloat)), 4));
      }
      const gcd = (a, b) => { a = a < 0n ? -a : a; b = b < 0n ? -b : b; while (b) [a, b] = [b, a % b]; return a; };
      if (expr.startsWith('gcd:')) { const [a, b] = split2(expr.slice(4), ',').map(pyInt); return gcd(a, b).toString(); }
      if (expr.startsWith('lcm:')) { const [a, b] = split2(expr.slice(4), ',').map(pyInt); const g = gcd(a, b); const l = g === 0n ? 0n : (a * b) / g; return (l < 0n ? -l : l).toString(); }
      if (expr.startsWith('prime:')) { const n = Number(pyInt(expr.slice(6))); if (n <= 1) return 'no'; for (let d = 2; d * d <= n; d++) if (n % d === 0) return 'no'; return 'yes'; }
      if (expr.startsWith('rev:')) return [...expr.slice(4)].reverse().join('');
      if (expr.startsWith('fact:')) { const n = pyInt(expr.slice(5)); if (n < 0n || n > 20n) return null; let f = 1n; for (let i = 2n; i <= n; i++) f *= i; return f.toString(); }
      if (expr.startsWith('sqrt:')) { const v = pyFloat(expr.slice(5)); if (v < 0) return null; return fmtF(pyRound(Math.sqrt(v), 3)); }
      if (expr.startsWith('pchg:')) { const [a, b] = split2(expr.slice(5), ',').map(pyFloat); if (a === 0) return null; return fmtF(pyRound((b - a) / a * 100, 2)); }
      if (expr.startsWith('words:')) { const w = expr.slice(6).trim(); return String(w ? w.split(/\s+/).length : 0); }
      const r = parseExpr(expr.replace(/×/g, '*').replace(/÷/g, '/').replace(/,/g, '.'));
      return r.t === 'i' ? r.v.toString() : fmtNum(F(pyRound(r.v, 6)));
    } catch (e) {
      return null;
    }
  }
  function split2(s, sep) { const parts = s.split(sep); if (parts.length !== 2) throw new Error('split'); return parts; }

  // decode-time hook: feed it every token; it returns the token that MUST come next (forced result / <ECALC>) or null
  class CircuitExecutor {
    constructor(tok) { this.tok = tok; this.queue = []; this.calls = 0; this.failed = 0; }
    next(ids) {
      if (!this.queue.length && ids.length && ids[ids.length - 1] === EQ) {
        const s = ids.lastIndexOf(CALC);
        let res = s >= 0 ? evaluate(this.tok.decode(ids.slice(s + 1, -1), true)) : null;
        this.calls++; if (res === null) { this.failed++; res = '?'; }
        this.queue = [...this.tok.encode(res), ECALC];
      }
      return this.queue.length ? this.queue.shift() : null;
    }
  }

  Object.assign(Lib, { evaluate, CircuitExecutor, NQ_IDS: { USER, MODEL, EOT, THINK, ETHINK, CALC, EQ, ECALC, BOS, EOS } });
  if (typeof module !== 'undefined') module.exports = { evaluate, CircuitExecutor };
})();
