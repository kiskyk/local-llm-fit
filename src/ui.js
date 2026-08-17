import { classify, normalizeModel, reverseLookup } from './fit.js';

const hf = (path) => fetch(`/api/hf?path=${encodeURIComponent(path)}`).then((r) => {
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
});

// 一覧APIはファイルサイズを返さないため、モデルごとに /tree/main を追加で取得する
async function fetchModel(entry) {
  try {
    const [tree, detail] = await Promise.all([
      hf(`/api/models/${entry.modelId}/tree/main`),
      hf(`/api/models/${entry.modelId}`),
    ]);
    const model = normalizeModel({
      modelId: entry.modelId,
      tree,
      config: detail.config ?? {},
      gguf: detail.gguf,
    });
    model.downloads = entry.downloads ?? 0;
    if (model.files.length > 0 && model.totalBillions) return model;
  } catch {
    // 1件取れなくても一覧全体は出す
  }
  return null;
}

// 直列だと件数×2リクエストで待ちが長いので、少数ずつ並列にする
async function fetchModels(list) {
  const results = [];
  const CONCURRENCY = 8;
  for (let i = 0; i < list.length; i += CONCURRENCY) {
    const chunk = await Promise.all(list.slice(i, i + CONCURRENCY).map(fetchModel));
    results.push(...chunk.filter(Boolean));
  }
  return results;
}

export async function loadModels(limit = 50) {
  // pipeline_tagで絞らないと、DL数の多い音声認識・TTS・埋め込みモデルがLLMより上に並ぶ
  const list = await hf(`/api/models?pipeline_tag=text-generation&filter=gguf&sort=downloads&direction=-1&limit=${limit}`);
  return fetchModels(list);
}

// Hugging Face全体からキーワード検索する（一覧の50件に無いモデルも逆引きできるように）
export async function searchModels(query, limit = 10) {
  const q = encodeURIComponent(query);
  const list = await hf(`/api/models?search=${q}&filter=gguf&pipeline_tag=text-generation&sort=downloads&direction=-1&limit=${limit}`);
  return fetchModels(list);
}

const ORDER = ['comfortable', 'tight', 'lower-quant', 'offload', 'no'];
const LABEL = {
  comfortable: ['余裕あり', '#2f7d4f'],
  tight: ['ギリギリ', '#a8683a'],
  'lower-quant': ['量子化を下げれば動く', '#a8683a'],
  offload: ['オフロードなら動く', '#7c4a24'],
  no: ['動かない', '#93826d'],
};

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function amazonSearchUrl(gpu) {
  const tag = 'kiskyk-22';
  // Apple統合メモリは型番そのままだとAmazon検索に引っかからないため、searchで上書きする
  return `https://www.amazon.co.jp/s?k=${encodeURIComponent(gpu.search ?? gpu.name)}&tag=${tag}`;
}

// 選択中のGPUで「動かない」判定が出たモデルにも手が届く、VRAMの多いGPUを1つ提案する
function upgradeSuggestion(gpus, currentVramGB) {
  const bigger = gpus
    .filter((g) => g.vramGB > currentVramGB && !g.name.startsWith('Apple'))
    .sort((a, b) => a.vramGB - b.vramGB);
  return bigger[0] ?? null;
}

export function render(container, judged, upgrade = null) {
  // 分類順を優先し、同じ分類の中ではダウンロード数の多い（＝よく知られた）モデルを上にする
  judged.sort((a, b) =>
    (ORDER.indexOf(a.result.verdict) - ORDER.indexOf(b.result.verdict)) ||
    ((b.model.downloads ?? 0) - (a.model.downloads ?? 0)));
  const note = `<p class="note">この判定は推定値です。実測で確認できているのは RTX 5070 Ti (16GB) のみで、実際の使用量はコンテキスト長や実行環境によって変わります。</p>`;
  const upgradeHtml = upgrade
    ? `<p class="note">より大きなモデルを余裕をもって動かすなら: <a href="${amazonSearchUrl(upgrade)}" target="_blank" rel="noopener sponsored">${esc(upgrade.name)}（${upgrade.vramGB}GB）をAmazonで探す</a></p>`
    : '';
  container.innerHTML = note + judged.map(({ model, result }) => {
    const [text, color] = LABEL[result.verdict];
    const quant = result.quant ? `｜推奨 ${result.quant}` : '';
    const warn = result.warning ? `<div class="warn">${esc(result.warning)}</div>` : '';
    return `<div class="item">
      <span class="badge" style="background:${color};color:#fff8ee">${text}</span>
      <strong>${esc(model.name)}</strong>${quant}${warn}
    </div>`;
  }).join('') + upgradeHtml;
}

// 逆引き結果: 判定の良い順 → 同じ判定ならVRAMの小さい（＝最小十分な）GPUを上に
export function renderReverse(container, rows) {
  rows.sort((a, b) =>
    (ORDER.indexOf(a.result.verdict) - ORDER.indexOf(b.result.verdict)) ||
    (a.gpu.vramGB - b.gpu.vramGB));
  container.innerHTML = rows.map(({ gpu, result }) => {
    const [text, color] = LABEL[result.verdict];
    const quant = result.quant ? `｜推奨 ${result.quant}` : '';
    const warn = result.warning ? `<div class="warn">${esc(result.warning)}</div>` : '';
    const buy = result.verdict === 'no' ? '' :
      `｜<a href="${amazonSearchUrl(gpu)}" target="_blank" rel="noopener sponsored">Amazonで探す</a>`;
    return `<div class="item">
      <span class="badge" style="background:${color};color:#fff8ee">${text}</span>
      <strong>${esc(gpu.name)}（${gpu.vramGB}GB）</strong>${quant}${buy}${warn}
    </div>`;
  }).join('');
}

async function main() {
  const gpuSelect = document.getElementById('gpu');
  const gpus = await fetch('./data/gpus.json').then((r) => r.json());
  gpuSelect.innerHTML = gpus.map((g, i) => `<option value="${i}">${esc(g.name)}（${g.vramGB}GB）</option>`).join('');

  const button = document.getElementById('go');
  const container = document.getElementById('result');
  const modelSelect = document.getElementById('model');
  const revButton = document.getElementById('rev');
  const revContainer = document.getElementById('revResult');

  // 逆引きセレクトの中身。検索するとHFの検索結果に置き換わる
  let revModels = [];
  const fillModelSelect = (models, emptyMessage) => {
    revModels = models;
    modelSelect.innerHTML = models.length
      ? models.map((m, i) => `<option value="${i}">${esc(m.name)}</option>`).join('')
      : `<option>${emptyMessage}</option>`;
  };

  // 先読みしておく（逆引きのセレクトを埋め、判定ボタンの待ちも減らす）
  const modelsPromise = loadModels().then((models) => {
    fillModelSelect(models, '読み込み失敗');
    return models;
  });
  modelsPromise.catch(() => { modelSelect.innerHTML = '<option>読み込み失敗</option>'; });

  const searchInput = document.getElementById('msearch');
  const searchButton = document.getElementById('msbtn');
  const doSearch = async () => {
    const query = searchInput.value.trim();
    searchButton.disabled = true;
    searchButton.textContent = '検索中…';
    try {
      // 空欄ならダウンロード数順の一覧に戻す
      fillModelSelect(query ? await searchModels(query) : await modelsPromise, '該当なし');
    } catch (e) {
      modelSelect.innerHTML = '<option>検索失敗</option>';
      revModels = [];
    } finally {
      searchButton.disabled = false;
      searchButton.textContent = '検索';
    }
  };
  searchButton.addEventListener('click', doSearch);
  searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSearch(); });

  const contextLength = () => Number(document.getElementById('ctx').value);

  button.addEventListener('click', async () => {
    button.disabled = true;
    button.textContent = '取得中…';
    try {
      const models = await modelsPromise;
      const vramBytes = gpus[Number(gpuSelect.value)].vramGB * 1024 ** 3;
      const judged = models.map((model) => ({ model, result: classify({ vramBytes, contextLength: contextLength(), model }) }));
      render(container, judged, upgradeSuggestion(gpus, gpus[Number(gpuSelect.value)].vramGB));
    } catch (e) {
      container.innerHTML = `<p class="warn">モデル一覧の取得に失敗しました（${esc(String(e))}）。時間をおいて再読み込みしてください。</p>`;
    } finally {
      button.disabled = false;
      button.textContent = '判定する';
    }
  });

  revButton.addEventListener('click', async () => {
    revButton.disabled = true;
    try {
      await modelsPromise.catch(() => {});
      const model = revModels[Number(modelSelect.value)];
      if (model) renderReverse(revContainer, reverseLookup(gpus, model, contextLength()));
    } finally {
      revButton.disabled = false;
    }
  });
}

main();
