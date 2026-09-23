// QPACK1 container reader + tensor unpacking (mirrors model/qpack.py exactly).
// Works in browser (ArrayBuffer from fetch) and in Node (Buffer -> ArrayBuffer) for testing.
'use strict';

const MAGIC = new TextEncoder().encode('QPACK1\0\0'); // 8 bytes

function readContainer(buf) {
  const dv = new DataView(buf);
  const magic = new Uint8Array(buf, 0, 8);
  for (let i = 0; i < 8; i++) if (magic[i] !== MAGIC[i]) throw new Error('not a QPACK1 container');
  const count = dv.getUint32(8, true);
  const toc = {};
  for (let i = 0; i < count; i++) {
    const base = 16 + 72 * i;
    let name = '';
    for (let j = 0; j < 56; j++) { const c = dv.getUint8(base + j); if (c === 0) break; name += String.fromCharCode(c); }
    const off = Number(dv.getBigUint64(base + 56, true));
    const len = Number(dv.getBigUint64(base + 64, true));
    toc[name] = [off, len];
  }
  return {
    buf, toc,
    blob(name) { const [o, l] = this.toc[name]; return new Uint8Array(this.buf, o, l); },
    text(name) { return new TextDecoder('utf-8').decode(this.blob(name)); },
    json(name) { return JSON.parse(this.text(name)); },
  };
}

// unpack 5-trits-per-byte -> Int8Array of {-1,0,1}, matching qpack.py pack_tern5/unpack_tern5
const _LUT = (() => {
  const lut = new Int8Array(256 * 5);
  for (let byte = 0; byte < 243; byte++) {
    let v = byte;
    for (let k = 0; k < 5; k++) { lut[byte * 5 + k] = (v % 3) - 1; v = (v / 3) | 0; }
  }
  return lut;
})();

function unpackTern5(bytes, n) {
  const out = new Float32Array(n); // caller multiplies by alpha; kept as {-1,0,1} here
  let w = 0;
  for (let i = 0; i < bytes.length && w < n; i++) {
    const b = bytes[i];
    for (let k = 0; k < 5 && w < n; k++) out[w++] = _LUT[b * 5 + k];
  }
  return out;
}

function fp16ToFp32(u16) {
  const s = (u16 & 0x8000) >> 15, e = (u16 & 0x7c00) >> 10, f = u16 & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * f * Math.pow(2, -24);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function readF16Array(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const n = bytes.length / 2, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = fp16ToFp32(dv.getUint16(i * 2, true));
  return out;
}

function readF32Array(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const n = bytes.length / 4, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getFloat32(i * 4, true);
  return out;
}

// Load every tensor named in manifest.json into a { name -> {data: Float32Array, shape:[...]} } map,
// plus the combined [vocab, code_bits] fingerprint table (frozen codes + trainable mm_delta).
function loadTensors(c) {
  const man = c.json('manifest.json');
  const cfg = c.json('config.json');
  const t = {};
  let codesFlat = null, deltaFlat = null;
  for (const e of man.tensors) {
    const n = e.shape.reduce((a, b) => a * b, 1);
    let data;
    if (e.kind === 'tern5') {
      const trits = unpackTern5(c.blob(e.blob), n);
      const alpha = readF32Array(c.blob(e.alpha)); // one per row (shape[0] rows)
      const cols = e.shape[1];
      data = new Float32Array(n);
      for (let r = 0; r < e.shape[0]; r++) { const a = alpha[r]; for (let cIdx = 0; cIdx < cols; cIdx++) data[r * cols + cIdx] = trits[r * cols + cIdx] * a; }
    } else if (e.kind === 'bits') {
      const packed = c.blob(e.blob); // shape [vocab, code_bits], MSB-first bits, 1 -> +1 else -1
      const cols = e.shape[1], rowBytes = Math.ceil(cols / 8);
      data = new Float32Array(n);
      for (let r = 0; r < e.shape[0]; r++) for (let bIdx = 0; bIdx < cols; bIdx++) {
        const byte = packed[r * rowBytes + (bIdx >> 3)]; const bit = (byte >> (7 - (bIdx & 7))) & 1;
        data[r * cols + bIdx] = bit ? 1 : -1;
      }
      codesFlat = { data, shape: e.shape };
      continue;
    } else if (e.kind === 'f16') {
      data = readF16Array(c.blob(e.blob));
      if (e.name.endsWith('mm_delta')) { deltaFlat = { data, shape: e.shape }; continue; }
    } else { data = readF32Array(c.blob(e.blob)); }
    t[e.name] = { data, shape: e.shape };
  }
  // combined table: codes with mm_delta added at rows [special_offset, special_offset+len(delta))
  const bits = cfg.code_bits, vocab = cfg.vocab_size;
  const table = codesFlat.data.slice();
  if (deltaFlat) {
    const start = cfg.special_offset;
    for (let r = 0; r < deltaFlat.shape[0]; r++) for (let cIdx = 0; cIdx < bits; cIdx++) table[(start + r) * bits + cIdx] += deltaFlat.data[r * bits + cIdx];
  }
  return { tensors: t, table, cfg, selftest: c.toc['selftest.json'] ? c.json('selftest.json') : null };
}

window.QOmniLib = window.QOmniLib || {};
Object.assign(window.QOmniLib, { readContainer, loadTensors, unpackTern5, fp16ToFp32, readF16Array, readF32Array });
