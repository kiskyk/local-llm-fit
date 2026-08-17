// src/fit.js

// 品質の低い順に並べる。添字がそのままrankになる。
const QUANT_ORDER = [
  'Q2_K', 'Q3_K_S', 'Q3_K_M', 'Q3_K_L',
  'Q4_K_S', 'Q4_K_M', 'Q5_K_S', 'Q5_K_M',
  'Q6_K', 'Q8_0',
];

export const QUALITY_FLOOR_RANK = QUANT_ORDER.indexOf('Q3_K_M');

export function parseQuant(filename) {
  const upper = filename.toUpperCase();
  // 長いラベルから先に照合する（Q3_K_M が Q3_K_S より前に一致しないように）
  const found = [...QUANT_ORDER]
    .sort((a, b) => b.length - a.length)
    .find((label) => upper.includes(label));
  if (!found) return null;
  return { label: found, rank: QUANT_ORDER.indexOf(found) };
}

const GB = 1024 ** 3;

// llama.cpp の計算バッファとフレームワークの占有分。brainの実測（16GB環境で
// 14.3GBのモデルが余裕0.9GB）と整合する値。
export const OVERHEAD_BYTES = 0.8 * GB;

// K と V の2本ぶんを、レイヤー数 × KVヘッド数 × ヘッド次元 × トークン数 で持つ。
// FP16 なので1要素2バイト。
export function kvCacheBytes(config, contextLength) {
  const headDim = config.hidden_size / config.num_attention_heads;
  const kvHeads = config.num_key_value_heads ?? config.num_attention_heads;
  return 2 * config.num_hidden_layers * kvHeads * headDim * contextLength * 2;
}

export function requiredBytes(fileBytes, config, contextLength) {
  return fileBytes + OVERHEAD_BYTES + kvCacheBytes(config, contextLength);
}

// 「35B-A3B」のように、総パラメータ数のあとに active 数が付く表記を読む。
const MOE_NAME_RE = /[-_]A(\d+(?:\.\d+)?)B\b/i;

export function detectMoE(modelName, config) {
  const matched = modelName.match(MOE_NAME_RE);
  if (matched) {
    return { isMoE: true, activeBillions: Number(matched[1]) };
  }
  const experts = config.num_experts ?? config.num_local_experts;
  if (experts && experts > 1) {
    return { isMoE: true, activeBillions: null };
  }
  return { isMoE: false, activeBillions: null };
}

// エキスパートをCPU側に置くと、GPUに載るのは active 相当の重みだけになる。
// 比率で概算し、KVキャッシュとオーバーヘッドは変わらないものとして足す。
export function offloadedBytes(fileBytes, totalBillions, activeBillions, config, contextLength) {
  const ratio = activeBillions / totalBillions;
  return fileBytes * ratio + OVERHEAD_BYTES + kvCacheBytes(config, contextLength);
}
