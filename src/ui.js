import { classify, normalizeModel } from './fit.js';

const hf = (path) => fetch(`/api/hf?path=${encodeURIComponent(path)}`).then((r) => {
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
});

// 一覧APIはファイルサイズを返さないため、モデルごとに /tree/main を追加で取得する。
// 直列だと50件×2リクエストで待ちが長いので、少数ずつ並列にする。
export async function loadModels(limit = 50) {
  const list = await hf(`/api/models?filter=gguf&sort=downloads&direction=-1&limit=${limit}`);
  const results = [];
  const CONCURRENCY = 8;
  for (let i = 0; i < list.length; i += CONCURRENCY) {
    const chunk = await Promise.all(list.slice(i, i + CONCURRENCY).map(async (entry) => {
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
        if (model.files.length > 0 && model.totalBillions) return model;
      } catch {
        // 1件取れなくても一覧全体は出す
      }
      return null;
    }));
    results.push(...chunk.filter(Boolean));
  }
  return results;
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

function amazonSearchUrl(gpuName) {
  const tag = 'ASSOCIATE_TAG'; // アソシエイトIDを取得したら差し替える
  return `https://www.amazon.co.jp/s?k=${encodeURIComponent(gpuName)}&tag=${tag}`;
}

// 選択中のGPUで「動かない」判定が出たモデルにも手が届く、VRAMの多いGPUを1つ提案する
function upgradeSuggestion(gpus, currentVramGB) {
  const bigger = gpus
    .filter((g) => g.vramGB > currentVramGB && !g.name.startsWith('Apple'))
    .sort((a, b) => a.vramGB - b.vramGB);
  return bigger[0] ?? null;
}

export function render(container, judged, upgrade = null) {
  judged.sort((a, b) => ORDER.indexOf(a.result.verdict) - ORDER.indexOf(b.result.verdict));
  const note = `<p class="note">この判定は推定値です。実測で確認できているのは RTX 5070 Ti (16GB) のみで、実際の使用量はコンテキスト長や実行環境によって変わります。</p>`;
  const upgradeHtml = upgrade
    ? `<p class="note">より大きなモデルを余裕をもって動かすなら: <a href="${amazonSearchUrl(upgrade.name)}" target="_blank" rel="noopener sponsored">${esc(upgrade.name)}（${upgrade.vramGB}GB）をAmazonで探す</a></p>`
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

async function main() {
  const gpuSelect = document.getElementById('gpu');
  const gpus = await fetch('./data/gpus.json').then((r) => r.json());
  gpuSelect.innerHTML = gpus.map((g, i) => `<option value="${i}">${esc(g.name)}（${g.vramGB}GB）</option>`).join('');

  const button = document.getElementById('go');
  const container = document.getElementById('result');
  let models = null;

  button.addEventListener('click', async () => {
    button.disabled = true;
    button.textContent = '取得中…';
    try {
      models ??= await loadModels();
      const vramBytes = gpus[Number(gpuSelect.value)].vramGB * 1024 ** 3;
      const contextLength = Number(document.getElementById('ctx').value);
      const judged = models.map((model) => ({ model, result: classify({ vramBytes, contextLength, model }) }));
      render(container, judged, upgradeSuggestion(gpus, gpus[Number(gpuSelect.value)].vramGB));
    } catch (e) {
      container.innerHTML = `<p class="warn">モデル一覧の取得に失敗しました（${esc(String(e))}）。時間をおいて再読み込みしてください。</p>`;
    } finally {
      button.disabled = false;
      button.textContent = '判定する';
    }
  });
}

main();
