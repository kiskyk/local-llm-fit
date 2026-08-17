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

import { classify } from './fit.js';

const GB2 = 1024 ** 3;
const CFG = { num_hidden_layers: 40, num_attention_heads: 32, num_key_value_heads: 8, hidden_size: 4096 };

function model(files, name = 'test-model-13B', totalBillions = 13) {
  return { name, config: CFG, totalBillions, files };
}

test('余裕があるときは comfortable', () => {
  const m = model([{ filename: 'a-Q4_K_M.gguf', sizeBytes: 8 * GB2 }]);
  const r = classify({ vramBytes: 16 * GB2, contextLength: 4096, model: m });
  assert.equal(r.verdict, 'comfortable');
  assert.equal(r.quant, 'Q4_K_M');
});

test('余裕が1GB未満なら tight', () => {
  const m = model([{ filename: 'a-Q4_K_M.gguf', sizeBytes: 14.5 * GB2 }]);
  const r = classify({ vramBytes: 16 * GB2, contextLength: 512, model: m });
  assert.equal(r.verdict, 'tight');
});

test('上位が入らないときは下の量子化を選ぶ', () => {
  const m = model([
    { filename: 'a-Q6_K.gguf', sizeBytes: 20 * GB2 },
    { filename: 'a-Q4_K_M.gguf', sizeBytes: 9 * GB2 },
  ]);
  const r = classify({ vramBytes: 16 * GB2, contextLength: 4096, model: m });
  assert.equal(r.quant, 'Q4_K_M');
  assert.equal(r.verdict, 'lower-quant');
});

test('Q3_K_M未満しか入らないときは警告を付ける', () => {
  const m = model([
    { filename: 'a-Q4_K_M.gguf', sizeBytes: 30 * GB2 },
    { filename: 'a-Q2_K.gguf', sizeBytes: 12 * GB2 },
  ]);
  const r = classify({ vramBytes: 16 * GB2, contextLength: 4096, model: m });
  assert.equal(r.quant, 'Q2_K');
  assert.ok(r.warning.includes('非推奨'));
});

test('MoEは入らなくてもオフロードなら動く', () => {
  const m = model([{ filename: 'a-Q4_K_M.gguf', sizeBytes: 20 * GB2 }], 'Qwen3.6-35B-A3B', 35);
  const r = classify({ vramBytes: 16 * GB2, contextLength: 4096, model: m });
  assert.equal(r.verdict, 'offload');
});

test('denseで到底入らないときは no', () => {
  const m = model([{ filename: 'a-Q2_K.gguf', sizeBytes: 60 * GB2 }], 'huge-70B', 70);
  const r = classify({ vramBytes: 8 * GB2, contextLength: 4096, model: m });
  assert.equal(r.verdict, 'no');
});

import { parseTotalBillions, normalizeModel, hasUsableConfig } from './fit.js';

test('モデル名から総パラメータ数を読む', () => {
  assert.equal(parseTotalBillions('gemma-3-27b-it-GGUF'), 27);
  assert.equal(parseTotalBillions('Qwen3.6-35B-A3B'), 35);
  assert.equal(parseTotalBillions('Llama-3.2-1B-Instruct'), 1);
  assert.equal(parseTotalBillions('some-model-without-size'), null);
});

test('HFの応答をclassifyが読める形に直す', () => {
  const result = normalizeModel({
    modelId: 'bartowski/gemma-3-27b-it-GGUF',
    tree: [
      { path: 'gemma-3-27b-it-Q4_K_M.gguf', size: 16000000000, type: 'file' },
      { path: 'README.md', size: 1000, type: 'file' },
      { path: 'imatrix', size: 0, type: 'directory' },
    ],
    config: { num_hidden_layers: 62 },
  });
  assert.equal(result.name, 'bartowski/gemma-3-27b-it-GGUF');
  assert.equal(result.totalBillions, 27);
  assert.equal(result.files.length, 1);
  assert.equal(result.files[0].filename, 'gemma-3-27b-it-Q4_K_M.gguf');
  assert.equal(result.files[0].sizeBytes, 16000000000);
});

test('KVキャッシュ計算に必要なconfigが揃っているか判定する', () => {
  assert.equal(hasUsableConfig({ num_hidden_layers: 62, num_attention_heads: 32, hidden_size: 5376 }), true);
  assert.equal(hasUsableConfig({ num_hidden_layers: 62 }), false);
  assert.equal(hasUsableConfig({}), false);
  assert.equal(hasUsableConfig(null), false);
});
