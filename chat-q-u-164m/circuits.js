// Exact circuits, ported from model/circuits.py. Same span protocol: <CALC> expr <EQ> [result] <ECALC>,
// forced by CircuitInterceptor exactly like the PyTorch CircuitLogitsProcessor forces it server-side.
'use strict';

// range(32768, 32778) -- Q architecture (Q-164M): tighter numbering, no image/audio ids to reserve space for in circuits.py
const USER = 32768, MODEL = 32769, EOT = 32770, THINK = 32771, ETHINK = 32772,
      CALC = 32773, EQ = 32774, ECALC = 32775, NEED = 32776, QUOTE = 32777;

const UNIT_FACTORS = {
  'km>mi': 0.621371, 'mi>km': 1.609344, 'kg>lb': 2.204623, 'lb>kg': 0.453592,
  'm>ft': 3.28084, 'ft>m': 0.3048, 'l>gal': 0.264172, 'gal>l': 3.78541,
  'cm>in': 0.393701, 'in>cm': 2.54,
};

function fmtNum(x) {
  if (typeof x === 'number' && !Number.isInteger(x)) {
    if (Math.abs(x - Math.round(x)) < 1e-9) return String(Math.round(x));
    return (Math.round(x * 10000) / 10000).toString();
  }
  return String(x);
}

// Arithmetic-only safe evaluator: digits, + - * / % ( ) . and unary minus, nothing else.
function safeArith(expr) {
  if (!/^[\d+\-*/%.() ]+$/.test(expr)) throw new Error('bad expression');
  // eslint-disable-next-line no-new-func
  const v = Function('"use strict"; return (' + expr + ')')();
  if (typeof v !== 'number' || !isFinite(v)) throw new Error('bad result');
  return v;
}

function daysInMonth(y, m) { return new Date(Date.UTC(y, m, 0)).getUTCDate(); }
function isoDate(d) { return d.toISOString().slice(0, 10); }
function addDays(iso, n) {
  const [y, m, d] = iso.split('-').map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d));
  dt.setUTCDate(dt.getUTCDate() + n);
  return isoDate(dt);
}
const WEEKDAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

function gcd(a, b) { a = Math.abs(a); b = Math.abs(b); while (b) { [a, b] = [b, a % b]; } return a; }
function lcm(a, b) { return Math.abs(a * b) / gcd(a, b); }
function isPrime(n) { if (n <= 1) return false; for (let d = 2; d * d <= n; d++) if (n % d === 0) return false; return true; }
function factorial(n) { let r = 1; for (let i = 2; i <= n; i++) r *= i; return r; }

// Mirrors circuits.py's evaluate(): returns a result STRING, or null if it cannot be computed.
function evaluate(expr) {
  try {
    if (expr.startsWith('date:')) {
      const [base, nStr] = expr.slice(5).split('+');
      return addDays(base, parseInt(nStr, 10));
    }
    if (expr.startsWith('wd:')) {
      const [y, m, d] = expr.slice(3).split('-').map(Number);
      return WEEKDAYS[new Date(Date.UTC(y, m - 1, d)).getUTCDay()];
    }
    if (expr.startsWith('dur:')) {
      const [d1, d2] = expr.slice(4).split(',');
      const toUtc = (s) => { const [y, m, d] = s.split('-').map(Number); return Date.UTC(y, m - 1, d); };
      return String(Math.round(Math.abs(toUtc(d2) - toUtc(d1)) / 86400000));
    }
    if (expr.startsWith('filter:')) {
      const [cond, rest] = expr.slice(7).split(/:(.*)/s);
      const ops = { '>=': (a, b) => a >= b, '<=': (a, b) => a <= b, '>': (a, b) => a > b, '<': (a, b) => a < b, '==': (a, b) => a === b };
      const sym = ['>=', '<=', '>', '<', '=='].find((s) => cond.startsWith(s));
      if (!sym) return null;
      const thr = parseFloat(cond.slice(sym.length));
      const xs = rest.split(',').filter((x) => ops[sym](parseFloat(x), thr));
      return xs.length ? xs.join(',') : 'none';
    }
    if (expr.startsWith('conv:')) {
      const rest = expr.slice(5);
      const sp = rest.indexOf(' ');
      const v = parseFloat(rest.slice(0, sp));
      const unitPart = rest.slice(sp + 1);
      if (unitPart === 'c>f') return fmtNum(Math.round((v * 9 / 5 + 32) * 100) / 100);
      if (unitPart === 'f>c') return fmtNum(Math.round((v - 32) * 5 / 9 * 100) / 100);
      if (UNIT_FACTORS[unitPart] !== undefined) return fmtNum(Math.round(v * UNIT_FACTORS[unitPart] * 100) / 100);
      return null;
    }
    if (expr.startsWith('cmp:')) {
      const [a, b] = expr.slice(4).split(',').map(Number);
      return a > b ? 'first' : a < b ? 'second' : 'equal';
    }
    if (expr.startsWith('sort:')) {
      return expr.slice(5).split(',').sort((a, b) => parseFloat(a) - parseFloat(b)).join(',');
    }
    if (expr.startsWith('count:')) return String(expr.slice(6).length);
    for (const [k, fn] of [['mean:', (v) => v.reduce((a, b) => a + b, 0) / v.length],
                           ['max:', (v) => Math.max(...v)], ['min:', (v) => Math.min(...v)],
                           ['sum:', (v) => v.reduce((a, b) => a + b, 0)]]) {
      if (expr.startsWith(k)) return fmtNum(Math.round(fn(expr.slice(k.length).split(',').map(Number)) * 10000) / 10000);
    }
    if (expr.startsWith('gcd:')) { const [a, b] = expr.slice(4).split(',').map(Number); return String(gcd(a, b)); }
    if (expr.startsWith('lcm:')) { const [a, b] = expr.slice(4).split(',').map(Number); return String(lcm(a, b)); }
    if (expr.startsWith('prime:')) return isPrime(parseInt(expr.slice(6), 10)) ? 'yes' : 'no';
    if (expr.startsWith('rev:')) return expr.slice(4).split('').reverse().join('');
    if (expr.startsWith('fact:')) { const n = parseInt(expr.slice(5), 10); return (n >= 0 && n <= 20) ? String(factorial(n)) : null; }
    if (expr.startsWith('sqrt:')) return fmtNum(Math.round(Math.sqrt(parseFloat(expr.slice(5))) * 1000) / 1000);
    if (expr.startsWith('pchg:')) { const [a, b] = expr.slice(5).split(',').map(Number); return fmtNum(Math.round((b - a) / a * 100 * 100) / 100); }
    if (expr.startsWith('words:')) return String(expr.slice(6).trim().split(/\s+/).filter(Boolean).length);
    return fmtNum(safeArith(expr));
  } catch (e) {
    return null;
  }
}

// Mirrors CircuitLogitsProcessor: watches the generated id stream; once <EQ> is emitted, evaluates the
// expression between the last <CALC> and that <EQ>, and queues the exact result chars + <ECALC> to be
// force-emitted in place of whatever the model would have sampled on its own.
class CircuitInterceptor {
  constructor(tok) { this.tok = tok; this.queue = []; this.handled = -1; }

  // call after each real (non-forced) token is appended to `ids` (the full generated sequence so far,
  // NOT including the prompt is fine as long as it includes at least the last <CALC>).
  afterToken(ids) {
    if (this.queue.length === 0 && ids.length && ids[ids.length - 1] === EQ && ids.length - 1 !== this.handled) {
      this.handled = ids.length - 1;
      let s = -1;
      for (let i = ids.length - 1; i >= 0; i--) if (ids[i] === CALC) { s = i; break; }
      if (s !== -1) {
        const expr = this.tok.decode(ids.slice(s + 1, ids.length - 1));
        const res = evaluate(expr);
        if (res !== null) {
          this.queue = [...this.charIds(res), ECALC];
        }
      }
    }
  }

  // next forced id, or null if nothing queued right now
  next() { return this.queue.length ? this.queue.shift() : null; }

  charIds(s) {
    const out = [];
    for (const ch of s) {
      const id = this.tok.vocab[ch];
      out.push(id !== undefined ? id : this.tok.encode(ch)[0]);
    }
    return out;
  }
}

window.QOmniLib = window.QOmniLib || {};
Object.assign(window.QOmniLib, { USER, MODEL, EOT, THINK, ETHINK, CALC, EQ, ECALC, NEED, QUOTE, evaluate, CircuitInterceptor });
