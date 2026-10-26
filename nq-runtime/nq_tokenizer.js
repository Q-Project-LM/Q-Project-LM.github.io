// N-Q tokenizer in the browser: the Hugging Face `tokenizers` pipeline of nq/tokenizer/tokenizer.json, i.e.
//   pre-tokenizer Sequence[ Digits(individual_digits=true), ByteLevel(add_prefix_space=false, GPT-2 regex) ] -> BPE
// Builds on tokenizer.js's ByteLevelBPE (byte maps, merges, decode); only the digit split is new.
// Checked id-for-id against the Python tokenizer on real chat data (web_tests/test_tokenizer.cjs).
'use strict';

(function () {
  const Lib = (typeof window !== 'undefined' ? (window.QOmniLib = window.QOmniLib || {}) : (globalThis.QOmniLib = globalThis.QOmniLib || {}));
  const SPLIT_RE = /'s|'t|'re|'ve|'m|'ll|'d| ?[^\s\p{L}\p{N}]+| ?\p{L}+| ?\p{N}+|\s+(?!\S)|\s+/gu;
  const DIGIT_RE = /(\p{N})/u;                        // tokenizers' Digits splits on Rust char::is_numeric = Unicode category N (Nd, Nl, No: '٣', 'Ⅻ', '①', '²')

  class NQTokenizer extends Lib.ByteLevelBPE {
    constructor(tj) {
      super(tj);   // added (special) tokens are matched in the raw text first, like `tokenizers` does
      const esc = [...this.specials.keys()].sort((a, b) => b.length - a.length).map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
      this.specialRe = new RegExp(`(${esc.join('|')})`);
    }

    encode(text) {
      const ids = [];
      for (const chunk of text.split(this.specialRe)) {
        if (!chunk) continue;
        if (this.specials.has(chunk)) ids.push(this.specials.get(chunk)); else this._encodeText(chunk, ids);
      }
      return ids;
    }

    _encodeText(text, ids) {
      for (const part of text.split(DIGIT_RE)) {   // capture group keeps the digits as separate parts
        if (!part) continue;
        for (const piece of part.match(SPLIT_RE) || []) {
          let mapped = '';
          for (const b of new TextEncoder().encode(piece)) mapped += this.byteToChar.get(b);
          for (const t of this._bpe(mapped)) {
            const id = this.vocab[t];
            if (id === undefined) throw new Error('unknown BPE token: ' + JSON.stringify(t));
            ids.push(id);
          }
        }
      }
    }
  }

  Lib.NQTokenizer = NQTokenizer;
  if (typeof module !== 'undefined') module.exports = { NQTokenizer };
})();
