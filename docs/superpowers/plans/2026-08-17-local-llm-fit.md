# ローカルLLM適合チェッカー 実装計画

> **エージェント実行者へ:** タスク単位で進める。各ステップはチェックボックス（`- [ ]`）で追跡する。

**Goal:** GPUを選ぶと、実用品質で動かせるローカルLLMと推奨量子化を一覧で出すWebツールを作る

**Architecture:** 判定ロジックを純関数の`src/fit.js`に隔離し、`node --test`だけで検証する。モデル情報はHugging Face APIから取得し、Cloudflare Pages Functionが中継する。UIは外部リソースを読み込まない単一のHTML。ergochairと同じ構成。

**Tech Stack:** 素のJavaScript（ESモジュール）、node --test、Cloudflare Pages と Pages Functions、Hugging Face API

## Global Constraints

- 設計: `docs/superpowers/specs/2026-08-17-local-llm-fit-design.md`
- **オーバーヘッドは0.8GB固定**（`OVERHEAD_BYTES`）。brainの実測と整合する値
- **Q3_K_Mを実用下限とする**。これ未満しか収まらない場合は収まっていても「非推奨」を返す
- 判定結果は推定値であり、実測で確認できているのはRTX 5070 Ti 16GBのみ。UIに明記する
- APIキーや認証情報をブラウザに渡さない（ergochairと同じ中継構造）
- 価格の手動記載はしない（Amazonアソシエイトの規約違反になる）
- コミットメッセージは日本語

---

### Task 1: 量子化レベルの抽出と品質順位

**Files:**
- Create: `src/fit.js`
- Test: `src/fit.test.mjs`

**Interfaces:**
- Produces: `parseQuant(filename)` → `{ label: string, rank: number } | null`、`QUALITY_FLOOR_RANK`（Q3_K_Mのrank）

- [ ] **Step 1: 失敗するテストを書く**

```js
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
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `node --test src/fit.test.mjs`
Expected: FAIL（`Cannot find module './fit.js'`）

- [ ] **Step 3: 最小の実装を書く**

```js
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
```

- [ ] **Step 4: テストが通ることを確認**

Run: `node --test src/fit.test.mjs`
Expected: PASS（3 tests）

- [ ] **Step 5: コミット**

```bash
git add src/fit.js src/fit.test.mjs && git commit -m "量子化レベルの抽出と品質順位を追加"
```

---

### Task 2: 必要VRAMの計算

**Files:**
- Modify: `src/fit.js`
- Test: `src/fit.test.mjs`

**Interfaces:**
- Consumes: なし
- Produces: `kvCacheBytes(config, contextLength)` → `number`、`requiredBytes(fileBytes, config, contextLength)` → `number`、定数 `OVERHEAD_BYTES`

- [ ] **Step 1: 失敗するテストを書く**

`src/fit.test.mjs` の末尾に追加する。

```js
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

test('brainの実測と整合する（コンテキストを除いた分）', () => {
  // gemma-3-27b-it: 14.3GB、16GB環境で余裕0.9GB
  assert.ok(Math.abs((16 * GB - (14.3 * GB + OVERHEAD_BYTES)) - 0.9 * GB) < 0.01 * GB);
  // gpt-oss-20b: 15.1GB、16GB環境で余裕0.1GB
  assert.ok(Math.abs((16 * GB - (15.1 * GB + OVERHEAD_BYTES)) - 0.1 * GB) < 0.01 * GB);
});
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `node --test src/fit.test.mjs`
Expected: FAIL（`kvCacheBytes is not a function`）

- [ ] **Step 3: 実装を追加する**

`src/fit.js` の末尾に追加する。

```js
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
```

- [ ] **Step 4: テストが通ることを確認**

Run: `node --test src/fit.test.mjs`
Expected: PASS（7 tests）

- [ ] **Step 5: コミット**

```bash
git add src/fit.js src/fit.test.mjs && git commit -m "必要VRAMの計算を追加（実測との整合をテストで固定）"
```

---

### Task 3: MoEの判定とオフロード時の必要量

**Files:**
- Modify: `src/fit.js`
- Test: `src/fit.test.mjs`

**Interfaces:**
- Consumes: `requiredBytes`
- Produces: `detectMoE(modelName, config)` → `{ isMoE: boolean, activeBillions: number | null }`、`offloadedBytes(fileBytes, totalBillions, activeBillions, config, contextLength)` → `number`

- [ ] **Step 1: 失敗するテストを書く**

```js
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
  const GB = 1024 ** 3;
  const config = { num_hidden_layers: 48, num_attention_heads: 32, num_key_value_heads: 8, hidden_size: 4096 };
  const full = 20 * GB;
  const offloaded = offloadedBytes(full, 35, 3, config, 4096);
  assert.ok(offloaded < full);
  assert.ok(offloaded > OVERHEAD_BYTES);
});
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `node --test src/fit.test.mjs`
Expected: FAIL（`detectMoE is not a function`）

- [ ] **Step 3: 実装を追加する**

```js
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
```

- [ ] **Step 4: テストが通ることを確認**

Run: `node --test src/fit.test.mjs`
Expected: PASS（11 tests）

- [ ] **Step 5: コミット**

```bash
git add src/fit.js src/fit.test.mjs && git commit -m "MoEの判定とオフロード時の必要量を追加"
```

---

### Task 4: 5分類の判定

**Files:**
- Modify: `src/fit.js`
- Test: `src/fit.test.mjs`

**Interfaces:**
- Consumes: `parseQuant`, `requiredBytes`, `detectMoE`, `offloadedBytes`, `QUALITY_FLOOR_RANK`
- Produces: `classify({ vramBytes, contextLength, model })` → `{ verdict, quant, headroomBytes, warning }`。`verdict` は `'comfortable' | 'tight' | 'lower-quant' | 'offload' | 'no'` のいずれか。`model` は `{ name, config, totalBillions, files: [{ filename, sizeBytes }] }`

- [ ] **Step 1: 失敗するテストを書く**

```js
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
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `node --test src/fit.test.mjs`
Expected: FAIL（`classify is not a function`）

- [ ] **Step 3: 実装を追加する**

```js
const TIGHT_MARGIN_BYTES = 1 * GB;

export function classify({ vramBytes, contextLength, model }) {
  const candidates = model.files
    .map((f) => ({ ...f, quant: parseQuant(f.filename) }))
    .filter((f) => f.quant !== null)
    .sort((a, b) => b.quant.rank - a.quant.rank); // 品質の高い順

  const best = candidates[0] ?? null;

  for (const [index, file] of candidates.entries()) {
    const need = requiredBytes(file.sizeBytes, model.config, contextLength);
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
    const need = offloadedBytes(best.sizeBytes, model.totalBillions, moe.activeBillions, model.config, contextLength);
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
```

- [ ] **Step 4: テストが通ることを確認**

Run: `node --test src/fit.test.mjs`
Expected: PASS（17 tests）

- [ ] **Step 5: コミット**

```bash
git add src/fit.js src/fit.test.mjs && git commit -m "5分類の判定ロジックを追加"
```

---

### Task 5: Hugging Face応答の正規化

**Files:**
- Modify: `src/fit.js`
- Test: `src/fit.test.mjs`

**Interfaces:**
- Consumes: なし
- Produces: `parseTotalBillions(name)` → `number | null`、`normalizeModel({ modelId, tree, config })` → `{ name, config, totalBillions, files }`。`tree` は `/api/models/{id}/tree/main` の応答（`[{ path, size, type }]`）

**なぜ必要か:** `classify` が受け取る形と、Hugging Face APIが返す形は違う。APIの `siblings` はファイル名しか持たずサイズを含まないため、`/tree/main` を別に取る必要がある。

- [ ] **Step 1: 失敗するテストを書く**

```js
import { parseTotalBillions, normalizeModel } from './fit.js';

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
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `node --test src/fit.test.mjs`
Expected: FAIL（`parseTotalBillions is not a function`）

- [ ] **Step 3: 実装を追加する**

```js
// 「27b」「35B」のような表記を拾う。A3B（active数）は別に扱うため、
// 直前がハイフンやドットで、直後がBで終わるものだけを対象にする。
const SIZE_RE = /(?:^|[-_.])(\d+(?:\.\d+)?)B(?:[-_.]|$)/i;

export function parseTotalBillions(name) {
  // A3B のような active 表記は総数ではないので除外する
  const cleaned = name.replace(/[-_]A\d+(?:\.\d+)?B\b/i, '');
  const matched = cleaned.match(SIZE_RE);
  return matched ? Number(matched[1]) : null;
}

export function normalizeModel({ modelId, tree, config }) {
  const files = tree
    .filter((entry) => entry.type === 'file' && entry.path.toLowerCase().endsWith('.gguf'))
    .map((entry) => ({ filename: entry.path, sizeBytes: entry.size }));
  return {
    name: modelId,
    config,
    totalBillions: parseTotalBillions(modelId),
    files,
  };
}
```

- [ ] **Step 4: テストが通ることを確認**

Run: `node --test src/fit.test.mjs`
Expected: PASS（19 tests）

- [ ] **Step 5: コミット**

```bash
git add src/fit.js src/fit.test.mjs && git commit -m "Hugging Face応答の正規化を追加"
```

---

### Task 6: GPUデータとHugging Face APIの中継

**Files:**
- Create: `data/gpus.json`
- Create: `functions/api/hf.js`
- Create: `serve.py`
- Create: `.env.example`
- Create: `.gitignore`

**Interfaces:**
- Produces: `/api/hf?path=<HFのAPIパス>` が Hugging Face の応答をそのまま返す

- [ ] **Step 1: GPU一覧を作る**

```json
[
  { "name": "RTX 5090", "vramGB": 32 },
  { "name": "RTX 5080", "vramGB": 16 },
  { "name": "RTX 5070 Ti", "vramGB": 16 },
  { "name": "RTX 5070", "vramGB": 12 },
  { "name": "RTX 4090", "vramGB": 24 },
  { "name": "RTX 4080 SUPER", "vramGB": 16 },
  { "name": "RTX 4070 Ti SUPER", "vramGB": 16 },
  { "name": "RTX 4070", "vramGB": 12 },
  { "name": "RTX 4060 Ti (16GB)", "vramGB": 16 },
  { "name": "RTX 4060", "vramGB": 8 },
  { "name": "RTX 3090", "vramGB": 24 },
  { "name": "RTX 3080", "vramGB": 10 },
  { "name": "RTX 3060 (12GB)", "vramGB": 12 },
  { "name": "Apple M4 Pro (24GB統合)", "vramGB": 16 },
  { "name": "Apple M4 Max (48GB統合)", "vramGB": 36 }
]
```

Apple Siliconは統合メモリのうちGPUが使える上限を目安として入れる。

- [ ] **Step 2: .gitignore を作る**

```
.env
node_modules/
__pycache__/
```

- [ ] **Step 3: 中継のPages Functionを書く**

```js
// functions/api/hf.js
// Hugging Face API への中継。ブラウザからのCORSを避け、応答をキャッシュする。
const ALLOWED_PREFIXES = ['/api/models', '/api/models/'];

export async function onRequestGet(context) {
  const url = new URL(context.request.url);
  const path = url.searchParams.get('path') ?? '';
  if (!ALLOWED_PREFIXES.some((p) => path.startsWith(p))) {
    return new Response(JSON.stringify({ error: 'invalid path' }), { status: 400 });
  }
  const target = `https://huggingface.co${path}`;
  const res = await fetch(target, { headers: { 'User-Agent': 'local-llm-fit/0.1' } });
  return new Response(res.body, {
    status: res.status,
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'public, max-age=3600',
    },
  });
}
```

- [ ] **Step 4: ローカル開発用のサーバを書く**

```python
# serve.py — 静的配信と /api/hf の中継をローカルで再現する
import http.server, socketserver, urllib.parse, urllib.request, json

PORT = 8080

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/api/hf':
            params = urllib.parse.parse_qs(parsed.query)
            path = params.get('path', [''])[0]
            if not path.startswith('/api/models'):
                self.send_error(400, 'invalid path')
                return
            req = urllib.request.Request(
                'https://huggingface.co' + path,
                headers={'User-Agent': 'local-llm-fit/0.1'},
            )
            with urllib.request.urlopen(req) as res:
                body = res.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

with socketserver.TCPServer(('127.0.0.1', PORT), Handler) as httpd:
    print(f'http://127.0.0.1:{PORT}')
    httpd.serve_forever()
```

- [ ] **Step 5: 中継が動くことを確認**

```bash
python serve.py
```

別の端末で次を実行する。

```bash
curl -s "http://127.0.0.1:8080/api/hf?path=/api/models?search=gguf&limit=1" | head -c 200
```

Expected: JSONが返り、`modelId` を含む。400が返る場合は `path` の指定を見直す

- [ ] **Step 6: コミット**

```bash
git add data/gpus.json functions/api/hf.js serve.py .gitignore && git commit -m "GPU一覧とHugging Face APIの中継を追加"
```

---

### Task 7: UIと結果の描画

**Files:**
- Create: `index.html`

**Interfaces:**
- Consumes: `classify`, `normalizeModel`（`src/fit.js`）、`/api/hf`、`data/gpus.json`

- [ ] **Step 1: 画面の骨格を書く**

```html
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ローカルLLM適合チェッカー</title>
<style>
:root { --paper:#f5efe3; --card:#fffcf5; --ink:#43362a; --muted:#93826d; --wood:#a8683a; --line:#e6dbc7; }
body { font-family:"Hiragino Sans","Yu Gothic UI",system-ui,sans-serif; background:var(--paper); color:var(--ink); max-width:760px; margin:0 auto; padding:24px 16px 48px; line-height:1.7; }
select { font-size:1rem; padding:6px 10px; border:1px solid var(--line); border-radius:8px; background:#fff; }
button { font-size:1.05rem; font-weight:600; padding:12px 32px; margin-top:16px; border:none; border-radius:999px; background:var(--wood); color:#fff8ee; cursor:pointer; }
.note { font-size:.85rem; color:var(--muted); border-left:3px solid var(--line); padding-left:10px; margin:20px 0; }
.item { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 18px; margin-bottom:10px; }
.badge { display:inline-block; font-size:.75rem; border-radius:999px; padding:2px 12px; margin-right:8px; }
.warn { color:#b3402e; font-size:.85rem; margin-top:6px; }
</style>
</head>
<body>
<h1>ローカルLLM適合チェッカー</h1>
<p>GPUを選ぶと、実用品質で動かせるモデルと推奨量子化を表示します。</p>
<label>GPU <select id="gpu"></select></label>
<label>コンテキスト長 <select id="ctx">
  <option value="4096">4096</option><option value="8192">8192</option>
  <option value="16384">16384</option><option value="32768">32768</option>
</select></label>
<button id="go">判定する</button>
<div id="result"></div>
<footer class="note">当サイトはAmazonアソシエイト・プログラムの参加者です。</footer>
<script type="module" src="./src/ui.js"></script>
</body>
</html>
```

- [ ] **Step 2: モデル一覧とファイルサイズを取得する**

`src/ui.js` を作る。**一覧APIはファイルサイズを返さないため、モデルごとに `/tree/main` を追加で取得する**。

```js
import { classify, normalizeModel } from './fit.js';

const hf = (path) => fetch(`/api/hf?path=${encodeURIComponent(path)}`).then((r) => {
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
});

export async function loadModels(limit = 50) {
  const list = await hf(`/api/models?filter=gguf&sort=downloads&direction=-1&limit=${limit}`);
  const results = [];
  for (const entry of list) {
    try {
      const tree = await hf(`/api/models/${entry.modelId}/tree/main`);
      const config = await hf(`/api/models/${entry.modelId}`).then((d) => d.config ?? {});
      const model = normalizeModel({ modelId: entry.modelId, tree, config });
      if (model.files.length > 0 && model.totalBillions) results.push(model);
    } catch {
      // 1件取れなくても一覧全体は出す
    }
  }
  return results;
}
```

- [ ] **Step 3: 判定して描画する**

```js
const ORDER = ['comfortable', 'tight', 'lower-quant', 'offload', 'no'];
const LABEL = {
  comfortable: ['余裕あり', '#2f7d4f'],
  tight: ['ギリギリ', '#a8683a'],
  'lower-quant': ['量子化を下げれば動く', '#a8683a'],
  offload: ['オフロードなら動く', '#7c4a24'],
  no: ['動かない', '#93826d'],
};

export function render(container, judged) {
  judged.sort((a, b) => ORDER.indexOf(a.result.verdict) - ORDER.indexOf(b.result.verdict));
  container.innerHTML = judged.map(({ model, result }) => {
    const [text, color] = LABEL[result.verdict];
    const quant = result.quant ? `｜推奨 ${result.quant}` : '';
    const warn = result.warning ? `<div class="warn">${result.warning}</div>` : '';
    return `<div class="item">
      <span class="badge" style="background:${color};color:#fff8ee">${text}</span>
      <strong>${model.name}</strong>${quant}${warn}
    </div>`;
  }).join('');
}
```

`classify` の呼び出しは `judged = models.map((model) => ({ model, result: classify({ vramBytes, contextLength, model }) }))` で作る。

- [ ] **Step 4: 推定である旨を明記する**

結果一覧の冒頭に次を固定表示する。

```
この判定は推定値です。実測で確認できているのは RTX 5070 Ti (16GB) のみで、
実際の使用量はコンテキスト長や実行環境によって変わります。
```

- [ ] **Step 5: ブラウザで確認する**

`.claude/launch.json` に `local-llm-fit`（`python serve.py`、ポート8080）を追加し、preview_start で起動する。
GPUを RTX 5070 Ti、コンテキスト長を4096にして判定し、**gemma系27Bクラスが `tight` か `lower-quant` に入る**ことを確認する（brainの実測と整合する）。コンソールエラーがないことも確認する。

- [ ] **Step 6: コミット**

```bash
git add index.html src/ui.js .claude/launch.json && git commit -m "UIと結果の描画を追加"
```

---

### Task 8: Amazonリンクと公開

**Files:**
- Modify: `index.html`, `src/ui.js`
- Create: `README.md`, `LICENSE`

- [ ] **Step 1: アソシエイトリンクを組み込む**

判定結果に「このモデルを余裕をもって動かすGPU」を1つ提示し、Amazon検索へのリンクを置く。

```js
function amazonSearchUrl(gpuName) {
  const tag = 'ASSOCIATE_TAG'; // アソシエイトIDを取得したら差し替える
  return `https://www.amazon.co.jp/s?k=${encodeURIComponent(gpuName)}&tag=${tag}`;
}
```

**価格は表示しない**（PA-API以外から取得した価格の掲載は規約違反）。

- [ ] **Step 2: アソシエイト参加の明記を入れる**

フッターに次を置く。

```
当サイトはAmazonアソシエイト・プログラムの参加者です。
```

- [ ] **Step 3: READMEを書く**

解こうとした問題、判定の考え方（Q3_K_Mを実用下限とする理由、MoEのオフロード）、ローカル実行の手順、テストの実行方法を書く。

- [ ] **Step 4: LICENSE（MIT、`Copyright (c) 2026 kiskyk`）を作る**

- [ ] **Step 5: テストを通してからコミット**

```bash
node --test src/fit.test.mjs && git add -A && git commit -m "Amazonリンクと公開用のREADMEを追加"
```

- [ ] **Step 6: 公開する**

```bash
gh repo create kiskyk/local-llm-fit --public --source=. --push --description "GPUから実用品質で動かせるローカルLLMを判定するツール"
```

Cloudflare Pages への接続はユーザーが操作する（GitHubリポジトリを指定してデプロイ）。

---

## 完了条件

- `node --test src/fit.test.mjs` が全件パスする
- RTX 5070 Ti (16GB) の判定が brain の実測と矛盾しない
- 公開URLでモデル一覧が表示され、コンソールエラーがない
- 価格を表示していない。アソシエイト参加の明記がある
