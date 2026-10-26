// Minimal byte-level BPE tokenizer (GPT-2 style), reading a standard Hugging Face `tokenizers` BPE file directly.
// No external dependency: same algorithm as `tokenizers.pre_tokenizers.ByteLevel` + `models.BPE` + `decoders.ByteLevel`.
'use strict';

function byteLevelMaps() {
  const bs = []; for (let i = 33; i <= 126; i++) bs.push(i); for (let i = 161; i <= 172; i++) bs.push(i); for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice(); let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const byteToChar = new Map(), charToByte = new Map();
  for (let i = 0; i < bs.length; i++) { byteToChar.set(bs[i], String.fromCodePoint(cs[i])); charToByte.set(cs[i], bs[i]); }
  return { byteToChar, charToByte };
}

// GPT-2 pre-tokenizer regex (matches `tokenizers`' default ByteLevel split pattern).
const SPLIT_RE = /'s|'t|'re|'ve|'m|'ll|'d| ?[^\s\p{L}\p{N}]+| ?\p{L}+| ?\p{N}+|\s+(?!\S)|\s+/gu;

class ByteLevelBPE {
  constructor(tokenizerJson) {
    const { byteToChar, charToByte } = byteLevelMaps(); this.byteToChar = byteToChar; this.charToByte = charToByte;
    const m = tokenizerJson.model;
    this.vocab = m.vocab; // token(string) -> id
    this.idToTok = new Array(Object.keys(this.vocab).length + (tokenizerJson.added_tokens ? tokenizerJson.added_tokens.length : 0));
    for (const [tokStr, id] of Object.entries(this.vocab)) this.idToTok[id] = tokStr;
    this.ranks = new Map(); m.merges.forEach((pair, i) => { const [a, b] = Array.isArray(pair) ? pair : pair.split(' '); this.ranks.set(a + '\u0000' + b, i); });
    this.specials = new Map(); this.idToSpecial = new Map();
    for (const t of (tokenizerJson.added_tokens || [])) { this.specials.set(t.content, t.id); this.idToSpecial.set(t.id, t.content); this.idToTok[t.id] = t.content; }
    this.bosId = this.specials.get('<bos>');
  }

  _bpe(token) { // token: string of byte-mapped chars (no spaces)
    let word = Array.from(token);
    if (word.length < 2) return word;
    while (true) {
      let best = null, bestRank = Infinity;
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.ranks.get(word[i] + '\u0000' + word[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best === null) break;
      const merged = word[best] + word[best + 1];
      word = word.slice(0, best).concat([merged], word.slice(best + 2));
    }
    return word;
  }

  encode(text) {
    const ids = [];
    const pieces = text.match(SPLIT_RE) || [];
    for (const piece of pieces) {
      let mapped = '';
      const bytes = new TextEncoder().encode(piece);
      for (const b of bytes) mapped += this.byteToChar.get(b);
      for (const tok of this._bpe(mapped)) {
        const id = this.vocab[tok];
        if (id === undefined) throw new Error('unknown BPE token: ' + JSON.stringify(tok));
        ids.push(id);
      }
    }
    return ids;
  }

  encodeWithBos(text) { return this.bosId !== undefined ? [this.bosId, ...this.encode(text)] : this.encode(text); }

  decode(ids, skipSpecial = true) {
    let mapped = '';
    for (const id of ids) {
      if (this.idToSpecial.has(id)) { if (!skipSpecial) mapped += this.idToSpecial.get(id); continue; }
      mapped += this.idToTok[id] || '';
    }
    const bytes = []; for (const ch of mapped) bytes.push(this.charToByte.get(ch.codePointAt(0)));
    return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
  }
}



window.QOmniLib = window.QOmniLib || {};
window.QOmniLib.ByteLevelBPE = ByteLevelBPE;
