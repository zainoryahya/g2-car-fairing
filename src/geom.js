// ===== B-spline car body geometry (pure math, no three.js) =====
const BASE = {
  L: 4.60, W: 1.82, axF: 0.95, axR: 3.70, wheelR: 0.33,
  kx: [0.05, 0.40, 0.95, 1.45, 1.75, 2.10, 2.55, 3.00, 3.40, 3.80, 4.20, 4.55],
  s:  [0.66, 0.76, 0.84, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.95, 0.91],
  h:  [0.70, 0.80, 0.87, 0.92, 1.08, 1.35, 1.44, 1.42, 1.27, 1.03, 0.99, 0.94],
  t:  [0.74, 0.78, 0.78, 0.76, 0.70, 0.64, 0.62, 0.62, 0.66, 0.76, 0.78, 0.76],
  w:  [0.86, 0.89, 0.91, 0.90, 0.90, 0.90, 0.90, 0.90, 0.91, 0.91, 0.89, 0.85],
  c:  [0.01, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01],
  fastH: [1.36, 1.20, 1.04], fastDx: 0.12
};

function naturalSpline(xs, ys) {
  const n = xs.length, h = [], al = [0], l = [1], mu = [0], z = [0], c = new Array(n).fill(0), b = [], d = [];
  for (let i = 0; i < n - 1; i++) h.push(xs[i + 1] - xs[i]);
  for (let i = 1; i < n - 1; i++) al.push(3 / h[i] * (ys[i + 1] - ys[i]) - 3 / h[i - 1] * (ys[i] - ys[i - 1]));
  for (let i = 1; i < n - 1; i++) { l[i] = 2 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]; mu[i] = h[i] / l[i]; z[i] = (al[i] - h[i - 1] * z[i - 1]) / l[i]; }
  for (let j = n - 2; j >= 0; j--) { c[j] = z[j] - mu[j] * c[j + 1]; b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3; d[j] = (c[j + 1] - c[j]) / (3 * h[j]); }
  return function (x) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] > x) hi = m; else lo = m; }
    const dx = x - xs[lo];
    return ys[lo] + b[lo] * dx + c[lo] * dx * dx + d[lo] * dx * dx * dx;
  };
}

// Build design (profiles + landmarks) from user parameters
function buildDesign(prm) {
  const sx = prm.L / BASE.L, sw = prm.W / BASE.W;
  const kx = BASE.kx.map(x => x * sx);
  kx[4] += prm.rake * 0.5; kx[5] += prm.rake;           // A-pillar / windshield rake
  kx[9] += prm.rear * BASE.fastDx * sx;                 // fastback pushes glass rearward
  const s = BASE.s.slice();
  const roofF = (prm.H - 0.92) / (1.44 - 0.92);
  const h = BASE.h.map((v, i) => (v - s[i] > 0.1 ? s[i] + (v - s[i]) * roofF : v));
  for (let k = 0; k < 3; k++) {
    const i = 8 + k, sed = h[i];
    const fb = s[i] + (BASE.fastH[k] - s[i]) * roofF;
    h[i] = sed + (fb - sed) * prm.rear;
  }
  const axF = BASE.axF * sx, axR = BASE.axR * sx;
  const t = BASE.t.map(v => v * sw);
  const w = BASE.w.map((v, i) => v * sw);
  const S = naturalSpline(kx, s), Hh = naturalSpline(kx, h), T = naturalSpline(kx, t), Wb = naturalSpline(kx, w), C = naturalSpline(kx, BASE.c);
  const flare = x => prm.flare * (Math.exp(-(((x - axF) / 0.42) ** 2)) + Math.exp(-(((x - axR) / 0.42) ** 2)));
  const Ra = 0.44, yc = 0.30, zb0 = 0.22;
  const zb = x => {
    let z = zb0;
    for (const xa of [axF, axR]) { const dx = x - xa; if (Math.abs(dx) < Ra) z = Math.max(z, yc + Math.sqrt(Ra * Ra - dx * dx)); }
    return z;
  };
  return {
    prm, kx, axF, axR, L: prm.L, sw,
    S, H: Hh, T, Wd: x => Wb(x) + flare(x), C, zb,
    land: { xA0: kx[3] + 0.04, xRoofF: kx[5] + 0.02, xRoofR: kx[7] + 0.02 + prm.rear * 0.15, xE: kx[9] + 0.06,
            xB: (kx[5] + kx[7]) / 2 + 0.06, xC: kx[8] + 0.02 + prm.rear * 0.1 }
  };
}

// Control net: rows along length (u), 13 columns across (v)
function buildNet(D, nStations) {
  const L = D.L, sw = D.sw;
  const rows = [];
  const section = (x, w, zb, s, h, t, c, rnd) => {
    const half = [
      [0.965 * w, zb],
      [w, zb + 0.55 * (s - zb)],
      [0.995 * w, s],
      [t + 0.5 * (w - t), s + 0.35 * (h - s)],
      [t, h],
      [0.5 * t, h + 0.7 * c]
    ];
    const row = [];
    for (let j = 0; j < 13; j++) {
      let lat, y;
      if (j === 6) { lat = 0; y = h + c; }
      else if (j < 6) { lat = -half[j][0]; y = half[j][1]; }
      else { lat = half[12 - j][0]; y = half[12 - j][1]; }
      const f = (lat / w) ** 2;
      row.push([x + rnd * f, y, lat]);
    }
    return row;
  };
  const fr = x => 0.26 * Math.exp(-x / 0.28);
  const rr = x => -0.24 * Math.exp(-(L - x) / 0.28);
  rows.push(section(0.10, 0.80 * sw, 0.22, 0.23, 0.24, 0.70 * sw, 0, 0.30));
  rows.push(section(0.00, 0.84 * sw, 0.22, 0.44, 0.50, 0.72 * sw, 0, 0.30));
  const x0 = D.kx[0], x1 = L - 0.05;
  for (let i = 0; i < nStations; i++) {
    const x = x0 + (x1 - x0) * i / (nStations - 1);
    rows.push(section(x, D.Wd(x), D.zb(x), D.S(x), D.H(x), D.T(x), D.C(x), fr(x) + rr(x)));
  }
  rows.push(section(L, 0.82 * sw, 0.26, 0.60, 0.64, 0.72 * sw, 0, -0.26));
  rows.push(section(L - 0.10, 0.78 * sw, 0.28, 0.29, 0.30, 0.68 * sw, 0, -0.26));
  return rows;
}

function clampedKnots(nC, p) {
  const U = [], spans = nC - p;
  for (let i = 0; i <= p; i++) U.push(0);
  for (let i = 1; i < spans; i++) U.push(i / spans);
  for (let i = 0; i <= p; i++) U.push(1);
  return U;
}
function findSpan(n, p, u, U) {
  if (u >= U[n + 1]) return n;
  if (u <= U[p]) return p;
  let lo = p, hi = n + 1, mid = (lo + hi) >> 1;
  while (u < U[mid] || u >= U[mid + 1]) { if (u < U[mid]) hi = mid; else lo = mid; mid = (lo + hi) >> 1; }
  return mid;
}
// NURBS Book A2.3
function dersBasis(i, u, p, nd, U) {
  const ndu = Array.from({ length: p + 1 }, () => new Float64Array(p + 1));
  const left = new Float64Array(p + 1), right = new Float64Array(p + 1);
  ndu[0][0] = 1;
  for (let j = 1; j <= p; j++) {
    left[j] = u - U[i + 1 - j]; right[j] = U[i + j] - u; let saved = 0;
    for (let r = 0; r < j; r++) {
      ndu[j][r] = right[r + 1] + left[j - r];
      const temp = ndu[r][j - 1] / ndu[j][r];
      ndu[r][j] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    ndu[j][j] = saved;
  }
  const ders = Array.from({ length: nd + 1 }, () => new Float64Array(p + 1));
  for (let j = 0; j <= p; j++) ders[0][j] = ndu[j][p];
  const a = [new Float64Array(p + 2), new Float64Array(p + 2)];
  for (let r = 0; r <= p; r++) {
    let s1 = 0, s2 = 1; a[0][0] = 1;
    for (let k = 1; k <= nd; k++) {
      let d = 0; const rk = r - k, pk = p - k;
      if (r >= k) { a[s2][0] = a[s1][0] / ndu[pk + 1][rk]; d = a[s2][0] * ndu[rk][pk]; }
      const j1 = rk >= -1 ? 1 : -rk;
      const j2 = (r - 1 <= pk) ? k - 1 : p - r;
      for (let j = j1; j <= j2; j++) { a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]; d += a[s2][j] * ndu[rk + j][pk]; }
      if (r <= pk) { a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]; d += a[s2][k] * ndu[r][pk]; }
      ders[k][r] = d;
      const tmp = s1; s1 = s2; s2 = tmp;
    }
  }
  let rr = p;
  for (let k = 1; k <= nd; k++) { for (let j = 0; j <= p; j++) ders[k][j] *= rr; rr *= (p - k); }
  return ders;
}

function makeSurface(net, p) {
  const m = net.length, n = net[0].length;
  const U = clampedKnots(m, p), V = clampedKnots(n, p);
  // side = -1 forces the left polynomial piece at a knot (one-sided limit from the left)
  const basisU = (u, side) => { let span = findSpan(m - 1, p, u, U); if (side < 0 && span > p && u === U[span]) span--; return { span, d: dersBasis(span, u, p, 2, U) }; };
  const basisV = (v, side) => { let span = findSpan(n - 1, p, v, V); if (side < 0 && span > p && v === V[span]) span--; return { span, d: dersBasis(span, v, p, 2, V) }; };
  const out = new Float64Array(18);
  const evalB = (bu, bv) => evalNet(net, p, bu, bv, out);
  const S = { net, p, m, n, U, V, basisU, basisV, evalB, orient: 1 };
  S.eval = (u, v) => evalB(basisU(u), basisV(v));
  S.geom = (o) => fundamentals(o, S.orient);
  const g = fundamentals(S.eval(0.5, 0.5), 1);
  S.orient = g[1] >= 0 ? 1 : -1;
  return S;
}

function evalNet(net, p, bu, bv, out) {
  out.fill(0);
  for (let a = 0; a <= p; a++) {
    const row = net[bu.span - p + a];
    const N0 = bu.d[0][a], N1 = bu.d[1][a], N2 = bu.d[2][a];
    for (let b = 0; b <= p; b++) {
      const P = row[bv.span - p + b], x = P[0], y = P[1], z = P[2];
      const M0 = bv.d[0][b], M1 = bv.d[1][b], M2 = bv.d[2][b];
      let w = N0 * M0; out[0] += w * x; out[1] += w * y; out[2] += w * z;
      w = N1 * M0; out[3] += w * x; out[4] += w * y; out[5] += w * z;
      w = N0 * M1; out[6] += w * x; out[7] += w * y; out[8] += w * z;
      w = N2 * M0; out[9] += w * x; out[10] += w * y; out[11] += w * z;
      w = N1 * M1; out[12] += w * x; out[13] += w * y; out[14] += w * z;
      w = N0 * M2; out[15] += w * x; out[16] += w * y; out[17] += w * z;
    }
  }
  return out;
}

function fundamentals(o, orient) {
  const sux = o[3], suy = o[4], suz = o[5], svx = o[6], svy = o[7], svz = o[8];
  let nx = suy * svz - suz * svy, ny = suz * svx - sux * svz, nz = sux * svy - suy * svx;
  const ln = Math.hypot(nx, ny, nz) || 1e-12; nx *= orient / ln; ny *= orient / ln; nz *= orient / ln;
  const E = sux * sux + suy * suy + suz * suz, F = sux * svx + suy * svy + suz * svz, G = svx * svx + svy * svy + svz * svz;
  const Lf = o[9] * nx + o[10] * ny + o[11] * nz, M = o[12] * nx + o[13] * ny + o[14] * nz, N = o[15] * nx + o[16] * ny + o[17] * nz;
  const den = (E * G - F * F) || 1e-12;
  const K = (Lf * N - M * M) / den;
  const H = -(E * N - 2 * F * M + G * Lf) / (2 * den);
  return [nx, ny, nz, K, H];
}

function tessellate(S, NU, NV) {
  const pos = new Float32Array(NU * NV * 3), nor = new Float32Array(NU * NV * 3), curv = new Float32Array(NU * NV * 2);
  const bvs = []; for (let j = 0; j < NV; j++) bvs.push(S.basisV(j / (NV - 1)));
  const absK = [], absH = [];
  for (let i = 0; i < NU; i++) {
    const bu = S.basisU(i / (NU - 1));
    for (let j = 0; j < NV; j++) {
      const o = S.evalB(bu, bvs[j]);
      const g = S.geom(o), k = i * NV + j;
      pos[3 * k] = o[0]; pos[3 * k + 1] = o[1]; pos[3 * k + 2] = o[2];
      nor[3 * k] = g[0]; nor[3 * k + 1] = g[1]; nor[3 * k + 2] = g[2];
      curv[2 * k] = g[3]; curv[2 * k + 1] = g[4];
      if (j % 3 === 0 && i % 3 === 0) { absK.push(Math.abs(g[3])); absH.push(Math.abs(g[4])); }
    }
  }
  const pct = (arr, q) => { arr.sort((a, b) => a - b); return arr[Math.floor(q * (arr.length - 1))] || 1; };
  const idx = new Uint32Array((NU - 1) * (NV - 1) * 6); let q = 0;
  for (let i = 0; i < NU - 1; i++) for (let j = 0; j < NV - 1; j++) {
    const a = i * NV + j, b = (i + 1) * NV + j, c = i * NV + j + 1, d = (i + 1) * NV + j + 1;
    if (S.orient > 0) { idx[q++] = a; idx[q++] = b; idx[q++] = c; idx[q++] = c; idx[q++] = b; idx[q++] = d; }
    else { idx[q++] = a; idx[q++] = c; idx[q++] = b; idx[q++] = c; idx[q++] = d; idx[q++] = b; }
  }
  return { pos, nor, curv, idx, kRange: pct(absK, 0.85), hRange: pct(absH, 0.85) };
}

function interiorKnots(U) { const r = []; for (const k of U) if (k > 0 && k < 1 && !r.includes(k)) r.push(k); return r; }

// Numerical G1/G2 check across all patch boundaries (knot lines)
function continuityCheck(S, hRef) {
  let dH = 0, dK = 0, dN = 0;
  const ks = 41;
  const test = (bu1, bv1, bu2, bv2) => {
    const g1 = S.geom(S.evalB(bu1, bv1)).slice(), g2 = S.geom(S.evalB(bu2, bv2));
    dN = Math.max(dN, Math.hypot(g1[0] - g2[0], g1[1] - g2[1], g1[2] - g2[2]));
    dH = Math.max(dH, Math.abs(g1[4] - g2[4]));
    dK = Math.max(dK, Math.abs(g1[3] - g2[3]));
  };
  for (const uk of interiorKnots(S.U)) {
    const b1 = S.basisU(uk, -1), b2 = S.basisU(uk, 1);
    for (let s = 1; s < ks; s++) { const bv = S.basisV(s / ks); test(b1, bv, b2, bv); }
  }
  for (const vk of interiorKnots(S.V)) {
    const b1 = S.basisV(vk, -1), b2 = S.basisV(vk, 1);
    for (let s = 1; s < ks; s++) { const bu = S.basisU(s / ks); test(bu, b1, bu, b2); }
  }
  return { dN, dH, dK, rel: dH / hRef };
}

// Porcupine comb along an isoparametric curve. dir: 'u' (v fixed) or 'v' (u fixed)
function comb(S, dir, c, samples) {
  const pts = [], tips = [], kap = [];
  for (let s = 0; s < samples; s++) {
    const t = s / (samples - 1);
    const o = dir === 'u' ? S.eval(t, c) : S.eval(c, t);
    const d1 = dir === 'u' ? [o[3], o[4], o[5]] : [o[6], o[7], o[8]];
    const d2 = dir === 'u' ? [o[9], o[10], o[11]] : [o[15], o[16], o[17]];
    const cr = [d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]];
    const l1 = Math.hypot(...d1), lc = Math.hypot(...cr);
    const k = lc / (l1 ** 3 || 1e-12);
    let nrm = [cr[1] * d1[2] - cr[2] * d1[1], cr[2] * d1[0] - cr[0] * d1[2], cr[0] * d1[1] - cr[1] * d1[0]];
    const ln = Math.hypot(...nrm) || 1; nrm = nrm.map(x => x / ln);
    pts.push([o[0], o[1], o[2]]); kap.push(k); tips.push(nrm);
  }
  const sorted = kap.slice().sort((a, b) => a - b), ref = sorted[Math.floor(0.92 * (sorted.length - 1))] || 1;
  const sc = 0.26 / ref;
  return pts.map((P, i) => { const l = Math.min(kap[i] * sc, 0.42); return { P, T: [P[0] - tips[i][0] * l, P[1] - tips[i][1] * l, P[2] - tips[i][2] * l] }; });
}

if (typeof module !== 'undefined') module.exports = { evalNet, fundamentals, clampedKnots, findSpan, dersBasis, BASE, buildDesign, buildNet, makeSurface, tessellate, continuityCheck, comb, interiorKnots };
