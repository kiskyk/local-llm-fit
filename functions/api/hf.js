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
