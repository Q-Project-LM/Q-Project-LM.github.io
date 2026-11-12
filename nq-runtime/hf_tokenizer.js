// A Hugging Face `tokenizers` reader for byte-level BPE tokenizers (Qwen2, Llama 3, SmolLM, GPT-2, N-Q, ...), driven by
// the model's own tokenizer.json: normalizer (NFC / NFKC / Sequence), pre-tokenizers (Split with a regex, Digits,
// ByteLevel, Sequence), BPE with merges (ignore_merges supported), added/special tokens, ByteLevel decoder.
// Checked id-for-id against the Python tokenizers library (tests/test_tokenizer.cjs).
'use strict';

(function () {
  const Lib = (typeof window !== 'undefined' ? (window.QPack = window.QPack || {}) : (globalThis.QPack = globalThis.QPack || {}));
  const GPT2_RE = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

  function byteMaps() {
    const bs = []; for (let i = 33; i <= 126; i++) bs.push(i); for (let i = 161; i <= 172; i++) bs.push(i); for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice(); let n = 0; for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
    const b2c = new Array(256), c2b = new Map(); bs.forEach((b, i) => { b2c[b] = String.fromCodePoint(cs[i]); c2b.set(String.fromCodePoint(cs[i]), b); });
    return { b2c, c2b };
  }

  // Rust regex -> JS: expand inline case-insensitive groups "(?i:...)" (no nesting) into explicit [xX] classes
  function rustToJsRegex(p) {
    return p.replace(/\(\?i:([^()]*)\)/g, (_, body) => '(?:' + body.replace(/[a-zA-Z]/g, (c) => `[${c.toLowerCase()}${c.toUpperCase()}]`) + ')');
  }

  class HFTokenizer {
    constructor(tj) {
      if (tj.model.type !== 'BPE') throw new Error(`tokenizer model ${tj.model.type} is not supported (byte-level BPE only)`);
      const { b2c, c2b } = byteMaps(); this.b2c = b2c; this.c2b = c2b;
      this.vocab = new Map(Object.entries(tj.model.vocab)); this.ignoreMerges = !!tj.model.ignore_merges;
      this.ranks = new Map(); tj.model.merges.forEach((m, i) => { const [a, b] = Array.isArray(m) ? m : m.split(' '); this.ranks.set(a + ' ' + b, i); });
      this.idToTok = []; for (const [t, id] of this.vocab) this.idToTok[id] = t;
      this.added = new Map(); this.specialIds = new Set();
      for (const t of tj.added_tokens || []) { this.added.set(t.content, t.id); this.idToTok[t.id] = t.content; if (t.special) this.specialIds.add(t.id); }
      const esc = [...this.added.keys()].sort((a, b) => b.length - a.length).map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
      this.addedRe = esc.length ? new RegExp(`(${esc.join('|')})`) : null;
      this.normalizers = this._norm(tj.normalizer); this.pre = this._pre(tj.pre_tokenizer);
      this.cache = new Map();
    }
    _norm(n) {
      if (!n) return [];
      if (n.type === 'Sequence') return n.normalizers.flatMap((x) => this._norm(x));
      if (n.type === 'NFC' || n.type === 'NFKC' || n.type === 'NFD' || n.type === 'NFKD') return [(s) => s.normalize(n.type)];
      if (n.type === 'Lowercase') return [(s) => s.toLowerCase()];
      throw new Error('normalizer ' + n.type + ' is not supported');
    }
    // pre-tokenizers as functions string[] -> string[] (byte-level mapping applied at the end)
    _pre(p) {
      if (!p) return [];
      if (p.type === 'Sequence') return p.pretokenizers.flatMap((x) => this._pre(x));
      if (p.type === 'Split') {
        const src = p.pattern.Regex != null ? rustToJsRegex(p.pattern.Regex) : p.pattern.String.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        if (p.behavior !== 'Isolated' || p.invert) throw new Error(`Split behavior ${p.behavior}${p.invert ? ' (inverted)' : ''} is not supported`);
        const re = new RegExp(src, 'gu');
        return [(parts) => parts.flatMap((s) => this._isolate(s, re))];
      }
      if (p.type === 'Digits') { const re = p.individual_digits ? /(\p{N})/u : /(\p{N}+)/u; return [(parts) => parts.flatMap((s) => s.split(re).filter(Boolean))]; }
      if (p.type === 'ByteLevel') {
        if (p.add_prefix_space) throw new Error('ByteLevel add_prefix_space is not supported');
        const re = new RegExp(GPT2_RE, 'gu');
        return p.use_regex === false ? [] : [(parts) => parts.flatMap((s) => s.match(re) || [])];
      }
      throw new Error('pre-tokenizer ' + p.type + ' is not supported');
    }
    _isolate(s, re) {   // matches become pieces, and so does the text between them
      const out = []; let last = 0; re.lastIndex = 0; let m;
      while ((m = re.exec(s)) !== null) { if (m.index > last) out.push(s.slice(last, m.index)); if (m[0]) out.push(m[0]); last = m.index + m[0].length; if (!m[0]) re.lastIndex++; }
      if (last < s.length) out.push(s.slice(last)); return out;
    }
    _bpe(word) {
      const hit = this.cache.get(word); if (hit) return hit;
      if (this.ignoreMerges && this.vocab.has(word)) { const r = [this.vocab.get(word)]; this.cache.set(word, r); return r; }
      let w = Array.from(word);
      while (w.length > 1) {
        let best = -1, br = Infinity;
        for (let i = 0; i < w.length - 1; i++) { const r = this.ranks.get(w[i] + ' ' + w[i + 1]); if (r !== undefined && r < br) { br = r; best = i; } }
        if (best < 0) break;
        const merged = w[best] + w[best + 1], nw = [];
        for (let i = 0; i < w.length; i++) { if (i < w.length - 1 && w[i] + ' ' + w[i + 1] === w[best] + ' ' + w[best + 1]) { nw.push(merged); i++; } else nw.push(w[i]); }
        w = nw;
      }
      const ids = w.map((t) => { const id = this.vocab.get(t); if (id === undefined) throw new Error('unknown BPE token ' + JSON.stringify(t)); return id; });
      if (this.cache.size < 100000) this.cache.set(word, ids);
      return ids;
    }
    encode(text) {
      const ids = [];
      for (const chunk of this.addedRe ? text.split(this.addedRe) : [text]) {
        if (!chunk) continue;
        if (this.added.has(chunk)) { ids.push(this.added.get(chunk)); continue; }
        let s = chunk; for (const f of this.normalizers) s = f(s);
        let parts = [s]; for (const f of this.pre) parts = f(parts);
        for (const piece of parts) {
          let mapped = ''; for (const b of new TextEncoder().encode(piece)) mapped += this.b2c[b];
          for (const id of this._bpe(mapped)) ids.push(id);
        }
      }
      return ids;
    }
    decode(ids, skipSpecial = true) {
      let out = '', bytes = [];
      const flush = () => { if (bytes.length) { out += new TextDecoder('utf-8').decode(new Uint8Array(bytes)); bytes = []; } };
      for (const id of ids) {
        const t = this.idToTok[id]; if (t === undefined) continue;
        if (this.added.has(t) && !this.vocab.has(t)) { if (!(skipSpecial && this.specialIds.has(id))) { flush(); out += t; } continue; }
        for (const ch of t) { const b = this.c2b.get(ch); if (b !== undefined) bytes.push(b); }
      }
      flush(); return out;
    }
  }

  Lib.HFTokenizer = HFTokenizer;
  if (typeof module !== 'undefined') module.exports = { HFTokenizer };
})();
