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
