// src/fit.test.mjs
import { test } from 'node:test';
import assert from 'node:assert';
import { parseQuant, QUALITY_FLOOR_RANK } from './fit.js';

test('ファイル名から量子化レベルを取り出す', () => {
  assert.equal(parseQuant('gemma-3-27b-it-Q4_K_M.gguf').label, 'Q4_K_M');
  assert.equal(parseQuant('model.Q3_K_M.gguf').label, 'Q3_K_M');
  assert.equal(parseQuant('model-q2_k.gguf').label, 'Q2_K');
  assert.equal(parseQuant('model-f16.gguf'), null);
});

test('品質の高い量子化ほどrankが大きい', () => {
  assert.ok(parseQuant('a-Q6_K.gguf').rank > parseQuant('a-Q4_K_M.gguf').rank);
  assert.ok(parseQuant('a-Q4_K_M.gguf').rank > parseQuant('a-Q3_K_M.gguf').rank);
  assert.ok(parseQuant('a-Q3_K_M.gguf').rank > parseQuant('a-Q2_K.gguf').rank);
});

test('実用下限はQ3_K_Mである', () => {
  assert.equal(parseQuant('a-Q3_K_M.gguf').rank, QUALITY_FLOOR_RANK);
  assert.ok(parseQuant('a-Q2_K.gguf').rank < QUALITY_FLOOR_RANK);
});

import { kvCacheBytes, requiredBytes, OVERHEAD_BYTES } from './fit.js';

const GB = 1024 ** 3;

// gemma-3-27b-it 相当の構成
const CONFIG = {
  num_hidden_layers: 62,
  num_attention_heads: 32,
  num_key_value_heads: 16,
  hidden_size: 5376,
};

test('オーバーヘッドは0.8GB', () => {
  assert.equal(OVERHEAD_BYTES, 0.8 * GB);
});

test('KVキャッシュはコンテキスト長に比例する', () => {
  const a = kvCacheBytes(CONFIG, 4096);
  const b = kvCacheBytes(CONFIG, 8192);
  assert.ok(a > 0);
  assert.equal(b, a * 2);
});

test('必要VRAMはファイルサイズとオーバーヘッドとKVキャッシュの合計', () => {
  const fileBytes = 14.3 * GB;
  const total = requiredBytes(fileBytes, CONFIG, 4096);
  assert.equal(total, fileBytes + OVERHEAD_BYTES + kvCacheBytes(CONFIG, 4096));
});

import { detectMoE, offloadedBytes } from './fit.js';

test('モデル名のA3B表記からMoEとactiveパラメータ数を読む', () => {
  const r = detectMoE('Qwen3.6-35B-A3B', {});
  assert.equal(r.isMoE, true);
  assert.equal(r.activeBillions, 3);
});

test('configのnum_expertsでもMoEと判定する', () => {
  assert.equal(detectMoE('some-model', { num_experts: 8 }).isMoE, true);
});

test('denseモデルはMoEではない', () => {
  assert.equal(detectMoE('gemma-3-27b-it', {}).isMoE, false);
});

test('オフロード時の必要量はactive比率で縮む', () => {
  const config = { num_hidden_layers: 48, num_attention_heads: 32, num_key_value_heads: 8, hidden_size: 4096 };
  const full = 20 * GB;
  const offloaded = offloadedBytes(full, 35, 3, config, 4096);
  assert.ok(offloaded < full);
  assert.ok(offloaded > OVERHEAD_BYTES);
});

test('brainの実測と整合する（コンテキストを除いた分）', () => {
  // gemma-3-27b-it: 14.3GB、16GB環境で余裕0.9GB
  assert.ok(Math.abs((16 * GB - (14.3 * GB + OVERHEAD_BYTES)) - 0.9 * GB) < 0.01 * GB);
  // gpt-oss-20b: 15.1GB、16GB環境で余裕0.1GB
  assert.ok(Math.abs((16 * GB - (15.1 * GB + OVERHEAD_BYTES)) - 0.1 * GB) < 0.01 * GB);
});
