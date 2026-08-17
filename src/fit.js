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
