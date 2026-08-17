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

const TIGHT_MARGIN_BYTES = 1 * GB;

// configが読めないモデル向けのKV概算。GQA世代の実構成（Llama-3-8B、gemma-3-27B、
// Qwen-32B）で1トークン・1Bパラメータあたり16〜24KBに収まるため、中間の20KBを使う。
const KV_FALLBACK_BYTES_PER_TOKEN_PER_B = 20e3;

// KVキャッシュ: 正確なconfigがあれば厳密式、なければパラメータ数からの概算、
// どちらも無ければ0（呼び出し側がパラメータ数不明のモデルを弾く前提）。
function modelKvBytes(model, contextLength) {
  if (hasUsableConfig(model.config)) return kvCacheBytes(model.config, contextLength);
  if (Number.isFinite(model.totalBillions)) {
    return KV_FALLBACK_BYTES_PER_TOKEN_PER_B * model.totalBillions * contextLength;
  }
  return 0;
}

export function classify({ vramBytes, contextLength, model }) {
  const kv = modelKvBytes(model, contextLength);
  const candidates = model.files
    .map((f) => ({ ...f, quant: parseQuant(f.filename) }))
    .filter((f) => f.quant !== null)
    .sort((a, b) => b.quant.rank - a.quant.rank); // 品質の高い順

  const best = candidates[0] ?? null;

  for (const [index, file] of candidates.entries()) {
    const need = file.sizeBytes + OVERHEAD_BYTES + kv;
    const headroomBytes = vramBytes - need;
    if (headroomBytes < 0) continue;

    const belowFloor = file.quant.rank < QUALITY_FLOOR_RANK;
    const warning = belowFloor
      ? `${file.quant.label} は実用下限の Q3_K_M を下回るため非推奨です`
      : '';

    // 最上位がそのまま入ったかどうかで comfortable / lower-quant を分ける
    const verdict = headroomBytes < TIGHT_MARGIN_BYTES
      ? 'tight'
      : index === 0
        ? 'comfortable'
        : 'lower-quant';

    return { verdict, quant: file.quant.label, headroomBytes, warning };
  }

  const moe = detectMoE(model.name, model.config);
  if (moe.isMoE && moe.activeBillions && best) {
    const need = best.sizeBytes * (moe.activeBillions / model.totalBillions) + OVERHEAD_BYTES + kv;
    if (need <= vramBytes) {
      return {
        verdict: 'offload',
        quant: best.quant.label,
        headroomBytes: vramBytes - need,
        warning: 'エキスパートをCPU側に置く構成（llama.cpp の --override-tensor）が必要です',
      };
    }
  }

  return { verdict: 'no', quant: null, headroomBytes: null, warning: '' };
}

// 逆引き: 1つのモデルを各GPUで判定する
export function reverseLookup(gpus, model, contextLength) {
  return gpus.map((gpu) => ({
    gpu,
    result: classify({ vramBytes: gpu.vramGB * GB, contextLength, model }),
  }));
}

// 「27b」「35B」のような表記を拾う。A3B（active数）は別に扱うため、
// 直前がハイフンやドットで、直後がBで終わるものだけを対象にする。
const SIZE_RE = /(?:^|[-_.])(\d+(?:\.\d+)?)B(?:[-_.]|$)/i;

export function parseTotalBillions(name) {
  // A3B のような active 表記は総数ではないので除外する
  const cleaned = name.replace(/[-_]A\d+(?:\.\d+)?B\b/i, '');
  const matched = cleaned.match(SIZE_RE);
  return matched ? Number(matched[1]) : null;
}

// GGUFリポジトリは config.json を持たないことが多い。KVキャッシュ計算に
// 必要なキーが欠けたまま classify に渡すと NaN 比較で誤判定するため、
// 呼び出し側はこれで事前に弾く。
export function hasUsableConfig(config) {
  return Boolean(
    config &&
    Number.isFinite(config.num_hidden_layers) &&
    Number.isFinite(config.num_attention_heads) &&
    Number.isFinite(config.hidden_size),
  );
}

export function normalizeModel({ modelId, tree, config, gguf }) {
  // 分割GGUF（-00001-of-00002 等）を1ファイルずつ独立に扱うと、断片のサイズで
  // 「収まる」と誤判定する。同じ量子化ラベルのファイルはサイズを合算して1つにする。
  const byLabel = new Map();
  const files = [];
  for (const entry of tree) {
    if (entry.type !== 'file' || !entry.path.toLowerCase().endsWith('.gguf')) continue;
    const quant = parseQuant(entry.path);
    const merged = quant && byLabel.get(quant.label);
    if (merged) {
      merged.sizeBytes += entry.size;
    } else {
      const file = { filename: entry.path, sizeBytes: entry.size };
      files.push(file);
      if (quant) byLabel.set(quant.label, file);
    }
  }
  // HF APIのggufメタデータが返す正確なパラメータ数を優先し、無ければ名前から読む
  const totalBillions = Number.isFinite(gguf?.total)
    ? gguf.total / 1e9
    : parseTotalBillions(modelId);
  return {
    name: modelId,
    config,
    totalBillions,
    files,
  };
}
