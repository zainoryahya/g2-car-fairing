// ===== Fairing optimisation (runs in a Web Worker; also testable in node) =====
function greville(U, p, i) { let s = 0; for (let k = 1; k <= p; k++) s += U[i + k]; return s / p; }

// Free variables: one scalar normal offset per control point (left/right mirrored).
// Hard points kept fixed: 3 rows at each end (bumpers) and the bottom column (rocker / wheel-arch edge).
function fairVars(net, p) {
  const S = makeSurface(net, p), m = net.length, vars = [];
  for (let i = 3; i <= m - 4; i++) for (let j = 1; j <= 6; j++) {
    const g = S.geom(S.eval(greville(S.U, p, i), greville(S.V, p, j)));
    const n = [g[0], g[1], j === 6 ? 0 : g[2]], l = Math.hypot(n[0], n[1], n[2]);
    vars.push({ i, j, n: [n[0] / l, n[1] / l, n[2] / l] });
  }
  return vars;
}
function applyOffsets(net, vars, t) {
  const N = net.map(r => r.map(q => q.slice()));
  for (let k = 0; k < vars.length; k++) {
    const v = vars[k], d = t[k], n = v.n;
    const P = N[v.i][v.j]; P[0] += d * n[0]; P[1] += d * n[1]; P[2] += d * n[2];
    if (v.j !== 6) { const Q = N[v.i][12 - v.j]; Q[0] += d * n[0]; Q[1] += d * n[1]; Q[2] -= d * n[2]; }
  }
  return N;
}

function quadGrid(m, n, p, NU, NV) {
  const U = clampedKnots(m, p), V = clampedKnots(n, p), bu = [], bv = [];
  const u0 = 0.07, u1 = 0.93, v0 = 0.07, v1 = 0.93;
  for (let i = 0; i < NU; i++) { const u = u0 + (u1 - u0) * i / (NU - 1), s = findSpan(m - 1, p, u, U); bu.push({ span: s, d: dersBasis(s, u, p, 2, U) }); }
  for (let j = 0; j < NV; j++) { const v = v0 + (v1 - v0) * j / (NV - 1), s = findSpan(n - 1, p, v, V); bv.push({ span: s, d: dersBasis(s, v, p, 2, V) }); }
  return { bu, bv, du: (u1 - u0) / (NU - 1), dv: (v1 - v0) / (NV - 1) };
}

// kind: 'var' = curvature variation  ∫ |∇H|² dA   (targets waviness)
//       'bend' = bending energy      ∫ (κ1²+κ2²) dA
function makeObjective(net0, p, vars, kind, NU = 64, NV = 32) {
  const m = net0.length, n = net0[0].length, Q = quadGrid(m, n, p, NU, NV);
  const orient = makeSurface(net0, p).orient;
  const out = new Float64Array(18), H = new Float64Array(NU * NV), Ar = new Float64Array(NU * NV), X = new Float64Array(NU * NV * 3), C = new Float64Array(NU * NV);
  function energy(net) {
    let Eb = 0;
    for (let i = 0; i < NU; i++) for (let j = 0; j < NV; j++) {
      evalNet(net, p, Q.bu[i], Q.bv[j], out);
      const g = fundamentals(out, orient), k = i * NV + j;
      const cx = out[4] * out[8] - out[5] * out[7], cy = out[5] * out[6] - out[3] * out[8], cz = out[3] * out[7] - out[4] * out[6];
      const a = Math.sqrt(cx * cx + cy * cy + cz * cz);
      H[k] = g[4]; Ar[k] = a; X[3 * k] = out[0]; X[3 * k + 1] = out[1]; X[3 * k + 2] = out[2];
      Eb += (4 * g[4] * g[4] - 2 * g[3]) * a;
    }
    if (kind === 'bend') return Eb * Q.du * Q.dv;
    let Ev = 0; C.fill(0);
    for (let i = 0; i < NU; i++) for (let j = 0; j < NV; j++) {
      const k = i * NV + j;
      if (i + 1 < NU) { const k2 = k + NV, d2 = (X[3 * k2] - X[3 * k]) ** 2 + (X[3 * k2 + 1] - X[3 * k + 1]) ** 2 + (X[3 * k2 + 2] - X[3 * k + 2]) ** 2; const e = (H[k2] - H[k]) ** 2 / (d2 + 1e-12) * Ar[k]; Ev += e; C[k] += e; }
      if (j + 1 < NV) { const k2 = k + 1, d2 = (X[3 * k2] - X[3 * k]) ** 2 + (X[3 * k2 + 1] - X[3 * k + 1]) ** 2 + (X[3 * k2 + 2] - X[3 * k + 2]) ** 2; const e = (H[k2] - H[k]) ** 2 / (d2 + 1e-12) * Ar[k]; Ev += e; C[k] += e; }
    }
    return Ev * Q.du * Q.dv;
  }
  const E0 = energy(net0);
  energy(net0); const C0 = C.slice();
  // local degradation: energy added where |∇H|² got worse, relative to J(0); 0 = nowhere worse
  const localUp = t => { energy(applyOffsets(net0, vars, t)); let s = 0; for (let k = 0; k < C.length; k++) s += Math.max(0, C[k] - C0[k]); return s * Q.du * Q.dv / E0; };
  const field = t => { energy(applyOffsets(net0, vars, t)); return { C: Array.from(C), C0: Array.from(C0), NU, NV }; };
  return { E0, energy, localUp, field, rel: t => energy(applyOffsets(net0, vars, t)) / E0 };
}

// ---------- linear thin-plate fairing: min ∫ |Suu|²+2|Suv|²+|Svv|²  + λ|t|², exact solve, λ by bisection so max|t| = δ ----------
function cholSolve(A, b, n) {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i * n + j]; for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
    if (i === j) { if (s <= 0) return null; L[i * n + i] = Math.sqrt(s); } else L[i * n + j] = s / L[j * n + j];
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
  return x;
}
function thinPlate(net0, p, vars, delta) {
  const m = net0.length, nC = net0[0].length, nv = vars.length, Q = quadGrid(m, nC, p, 72, 36);
  const map = new Map();
  vars.forEach((v, k) => { map.set(v.i * nC + v.j, [k, 1]); if (v.j !== 6) map.set(v.i * nC + 12 - v.j, [k, -1]); });
  const AtA = new Float64Array(nv * nv), Atb = new Float64Array(nv), out = new Float64Array(18);
  const coef = new Float64Array(nv * 3), touched = [];
  const terms = [[2, 0, 1, 9], [1, 1, Math.SQRT2, 12], [0, 2, 1, 15]]; // (du order, dv order, weight, offset in out)
  for (const bu of Q.bu) for (const bv of Q.bv) {
    evalNet(net0, p, bu, bv, out);
    for (const [ou, ov, w, off] of terms) {
      touched.length = 0;
      for (let a = 0; a <= p; a++) for (let b = 0; b <= p; b++) {
        const e = map.get((bu.span - p + a) * nC + (bv.span - p + b)); if (!e) continue;
        const [k, mz] = e, c = w * bu.d[ou][a] * bv.d[ov][b], nn = vars[k].n;
        if (coef[3 * k] === 0 && coef[3 * k + 1] === 0 && coef[3 * k + 2] === 0) touched.push(k);
        coef[3 * k] += c * nn[0]; coef[3 * k + 1] += c * nn[1]; coef[3 * k + 2] += c * nn[2] * mz;
      }
      for (let c = 0; c < 3; c++) {
        const bval = w * out[off + c];
        for (const k1 of touched) {
          const a1 = coef[3 * k1 + c]; if (!a1) continue;
          Atb[k1] += a1 * bval;
          for (const k2 of touched) AtA[k1 * nv + k2] += a1 * coef[3 * k2 + c];
        }
      }
      for (const k of touched) { coef[3 * k] = coef[3 * k + 1] = coef[3 * k + 2] = 0; }
    }
  }
  let tr = 0; for (let i = 0; i < nv; i++) tr += AtA[i * nv + i]; tr /= nv;
  const solve = lam => { const A = AtA.slice(); for (let i = 0; i < nv; i++) A[i * nv + i] += lam; return cholSolve(A, Atb.map(x => -x), nv); };
  const maxAbs = t => t.reduce((s, x) => Math.max(s, Math.abs(x)), 0);
  let lo = Math.log(tr * 1e-7), hi = Math.log(tr * 1e3), best = solve(Math.exp(hi));
  const t0 = solve(Math.exp(lo));
  if (t0 && maxAbs(t0) <= delta) return Array.from(t0);
  for (let it = 0; it < 40; it++) {
    const mid = (lo + hi) / 2, t = solve(Math.exp(mid));
    if (t && maxAbs(t) <= delta) { best = t; hi = mid; } else lo = mid;
  }
  return Array.from(best);
}

// ---------- metaheuristics on the nonlinear energy (box |t_k| ≤ δ) ----------
function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function initPop(NP, D, delta, R, seeds) {
  const pop = [];
  for (const s of seeds) pop.push(Float64Array.from(s));
  const centre = seeds.length > 1 ? seeds[1] : seeds[0];
  while (pop.length < NP) {
    const base = pop.length % 2 && seeds.length > 1 ? centre : seeds[0];
    const spread = seeds.length > 1 ? 0.15 : 0.3;
    pop.push(Float64Array.from({ length: D }, (_, j) => Math.max(-delta, Math.min(delta, base[j] + (2 * R() - 1) * spread * delta))));
  }
  return pop;
}
function runMeta(method, f, D, delta, budget, seeds, cb, seed = 7, shouldStop = () => false) {
  const R = rng(seed), NP = 30, pen = 0.05;
  const F = t => { let s = 0; for (let j = 0; j < D; j++) s += (t[j] / delta) ** 2; return f(t) + pen * s / D; };
  const pop = initPop(NP, D, delta, R, seeds), fit = pop.map(F);
  let evals = NP, bi = fit.indexOf(Math.min(...fit)), best = pop[bi].slice(), bestF = fit[bi];
  const hist = [[evals, bestF]];
  const clip = x => Math.max(-delta, Math.min(delta, x));
  const report = () => { hist.push([evals, bestF]); cb && cb(evals, bestF, best, hist); };
  if (method === 'jde') {
    const Fi = new Float64Array(NP).fill(0.5), CRi = new Float64Array(NP).fill(0.9), tr = new Float64Array(D);
    while (evals < budget && !shouldStop()) {
      for (let i = 0; i < NP && evals < budget; i++) {
        const Fn = R() < 0.1 ? 0.1 + 0.9 * R() : Fi[i], CRn = R() < 0.1 ? R() : CRi[i];
        let r1, r2, r3; do r1 = R() * NP | 0; while (r1 === i); do r2 = R() * NP | 0; while (r2 === i || r2 === r1); do r3 = R() * NP | 0; while (r3 === i || r3 === r1 || r3 === r2);
        const jr = R() * D | 0;
        for (let j = 0; j < D; j++) {
          if (R() < CRn || j === jr) { let x = pop[r1][j] + Fn * (pop[r2][j] - pop[r3][j]); if (x > delta) x = (delta + pop[i][j]) / 2; if (x < -delta) x = (-delta + pop[i][j]) / 2; tr[j] = x; }
          else tr[j] = pop[i][j];
        }
        const ft = F(tr); evals++;
        if (ft <= fit[i]) { pop[i].set(tr); fit[i] = ft; Fi[i] = Fn; CRi[i] = CRn; if (ft < bestF) { bestF = ft; best = tr.slice(); } }
      }
      report();
    }
  } else if (method === 'pso') {
    const w = 0.729, c1 = 1.49445, c2 = 1.49445, vmax = 0.25 * delta;
    const V = pop.map(() => new Float64Array(D)), pb = pop.map(x => x.slice()), pbf = fit.slice();
    while (evals < budget && !shouldStop()) {
      for (let i = 0; i < NP && evals < budget; i++) {
        const x = pop[i], v = V[i];
        for (let j = 0; j < D; j++) {
          v[j] = w * v[j] + c1 * R() * (pb[i][j] - x[j]) + c2 * R() * (best[j] - x[j]);
          v[j] = Math.max(-vmax, Math.min(vmax, v[j])); x[j] = clip(x[j] + v[j]);
        }
        const fx = F(x); evals++;
        if (fx < pbf[i]) { pbf[i] = fx; pb[i] = x.slice(); if (fx < bestF) { bestF = fx; best = x.slice(); } }
      }
      report();
    }
  } else if (method === 'gwo') {
    const maxIt = Math.ceil((budget - NP) / NP); let it = 0;
    const order = () => fit.map((f, i) => [f, i]).sort((a, b) => a[0] - b[0]).slice(0, 3).map(e => pop[e[1]].slice());
    let [A, B, Dl] = order();
    while (evals < budget && !shouldStop()) {
      const a = 2 - 2 * it / maxIt;
      for (let i = 0; i < NP && evals < budget; i++) {
        const x = pop[i];
        for (let j = 0; j < D; j++) {
          let s = 0;
          for (const L of [A, B, Dl]) { const a1 = 2 * a * R() - a, c1 = 2 * R(); s += L[j] - a1 * Math.abs(c1 * L[j] - x[j]); }
          x[j] = clip(s / 3);
        }
        fit[i] = F(x); evals++;
        if (fit[i] < bestF) { bestF = fit[i]; best = x.slice(); }
      }
      [A, B, Dl] = order(); it++;
      report();
    }
  }
  return { best, bestF, evals, hist };
}

// Reduced (smooth) design space: offset field t(i,j) = Σ c_ab B_a(i) B_b(j) with coarse cubic B-splines.
// Partition of unity ⇒ |c| ≤ δ guarantees |t| ≤ δ, and the field cannot carry control-point-level noise.
function reducedBasis(vars, Ka = 9, Kb = 4) {
  const is = vars.map(v => v.i), js = vars.map(v => v.j);
  const i0 = Math.min(...is), i1 = Math.max(...is), j0 = Math.min(...js), j1 = Math.max(...js);
  const basis = (K, s) => { const U = clampedKnots(K, 3), sp = findSpan(K - 1, 3, s, U), d = dersBasis(sp, s, 3, 0, U)[0], out = new Float64Array(K); for (let a = 0; a <= 3; a++) out[sp - 3 + a] = d[a]; return out; };
  return vars.map(v => {
    const bi = basis(Ka, (v.i - i0) / (i1 - i0)), bj = basis(Kb, (v.j - j0) / (j1 - j0)), row = new Float64Array(Ka * Kb);
    for (let a = 0; a < Ka; a++) for (let b = 0; b < Kb; b++) row[a * Kb + b] = bi[a] * bj[b];
    return row;
  });
}
function expand(Phi, c) { return Phi.map(r => { let s = 0; for (let k = 0; k < r.length; k++) s += r[k] * c[k]; return s; }); }
function project(Phi, t, delta) {
  const D = Phi[0].length, A = new Float64Array(D * D), b = new Float64Array(D);
  Phi.forEach((r, k) => { for (let a = 0; a < D; a++) { b[a] += r[a] * t[k]; for (let c = 0; c < D; c++) A[a * D + c] += r[a] * r[c]; } });
  for (let a = 0; a < D; a++) A[a * D + a] += 1e-9;
  return Array.from(cholSolve(A, b, D)).map(x => Math.max(-delta, Math.min(delta, x)));
}

if (typeof module !== 'undefined') module.exports = { reducedBasis, expand, project, fairVars, applyOffsets, makeObjective, thinPlate, runMeta };

// ---------- worker protocol ----------
if (typeof self !== 'undefined' && typeof window === 'undefined' && typeof module === 'undefined') {
  let stop = false;
  self.onmessage = (e) => {
    const msg = e.data;
    if (msg.cmd === 'stop') { stop = true; return; }
    if (msg.cmd !== 'run') return;
    stop = false;
    const { net0, p, method, kind, delta, budget, seed } = msg;
    const t0 = performance.now();
    const vars = fairVars(net0, p), D = vars.length;
    const objV = makeObjective(net0, p, vars, 'var'), objB = makeObjective(net0, p, vars, 'bend');
    const obj = kind === 'bend' ? objB : objV;
    const summary = (t) => ({ relVar: objV.rel(t), relBend: objB.rel(t), maxDev: t.reduce((s, x) => Math.max(s, Math.abs(x)), 0) });
    const reduced = msg.space !== 'full' && method !== 'tp';
    const Phi = reduced ? reducedBasis(vars) : null;
    const Dm = reduced ? Phi[0].length : D;
    const toT = c => reduced ? expand(Phi, c) : Array.from(c);
    const f = reduced ? (c => obj.rel(expand(Phi, c))) : obj.rel;
    let last = 0;
    const cb = (ev, fv, best, hist) => { const now = performance.now(); if (now - last > 350) { last = now; self.postMessage({ type: 'progress', vars, best: toT(best), evals: ev, f: fv, hist, Dm }); } };
    let result;
    if (method === 'tp') {
      const tp = thinPlate(net0, p, vars, delta);
      result = { best: tp, evals: 1, hist: [[0, 1], [1, obj.rel(tp)]] };
    } else {
      const seeds = [new Float64Array(Dm)];
      if (method === 'hyb') { const tp = thinPlate(net0, p, vars, delta); seeds.push(Float64Array.from(reduced ? project(Phi, tp, delta) : tp)); }
      result = runMeta(method === 'hyb' ? 'jde' : method, f, Dm, delta, budget, seeds, cb, seed, () => stop);
      result.best = toT(result.best);
    }
    const best = Array.from(result.best);
    self.postMessage({ type: 'done', vars, best, evals: result.evals, hist: result.hist, ms: performance.now() - t0, D, Dm, ...summary(best), stopped: stop });
  };
}
