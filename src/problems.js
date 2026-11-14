// Test problems: four car-body variants, each a clamped bicubic B-spline skin.
const G = require('./geom.js'); Object.assign(global, G);
const O = require('./opt.js');

const SURFACES = {
  S1: { name: 'Sedan (notchback)', design: { L: 4.60, W: 1.82, H: 1.44, rake: 0.00, rear: 0.0, flare: 0.025 }, stations: 28 },
  S2: { name: 'Fastback coupe',     design: { L: 4.70, W: 1.86, H: 1.38, rake: 0.12, rear: 1.0, flare: 0.040 }, stations: 28 },
  S3: { name: 'Tall hatch',         design: { L: 4.30, W: 1.80, H: 1.60, rake: -0.10, rear: 0.6, flare: 0.020 }, stations: 26 },
  S4: { name: 'Long GT',            design: { L: 4.95, W: 1.92, H: 1.32, rake: 0.22, rear: 0.8, flare: 0.060 }, stations: 32 }
};

const GRID_OPT = [64, 32], GRID_VAL = [128, 64];

function setup(sid, delta, p = 3) {
  const S = SURFACES[sid];
  const D = buildDesign(S.design), net = buildNet(D, S.stations);
  const vars = O.fairVars(net, p);
  const objOpt = O.makeObjective(net, p, vars, 'var', ...GRID_OPT);
  const objVal = O.makeObjective(net, p, vars, 'var', ...GRID_VAL);
  const objBend = O.makeObjective(net, p, vars, 'bend', ...GRID_VAL);
  const Phi = O.reducedBasis(vars);
  const tp = O.thinPlate(net, p, vars, delta);
  return { sid, delta, net, p, vars, objOpt, objVal, objBend, Phi, tp };
}

// RMS of second differences of the offset field over the control grid (mm)
function roughness(vars, t) {
  const key = new Map(); vars.forEach((v, k) => key.set(v.i + ',' + v.j, t[k]));
  let s = 0, n = 0;
  for (const v of vars) {
    const b = key.get(v.i + ',' + v.j);
    const a = key.get((v.i - 1) + ',' + v.j), c = key.get((v.i + 1) + ',' + v.j);
    if (a !== undefined && c !== undefined) { s += (a - 2 * b + c) ** 2; n++; }
    const d = key.get(v.i + ',' + (v.j - 1)), e = key.get(v.i + ',' + (v.j + 1));
    if (d !== undefined && e !== undefined) { s += (d - 2 * b + e) ** 2; n++; }
  }
  return Math.sqrt(s / n) * 1000;
}

function metrics(P, t) {
  return {
    J_opt: P.objOpt.rel(t), J_val: P.objVal.rel(t), J_up: P.objVal.localUp(t), J_bend: P.objBend.rel(t),
    maxdev_mm: t.reduce((s, x) => Math.max(s, Math.abs(x)), 0) * 1000, rough_mm: roughness(P.vars, t)
  };
}

// search-space objective (what the optimiser sees, incl. the small offset penalty used by runMeta)
function searchObjective(P, space) {
  const red = space === 'reduced', Dm = red ? P.Phi[0].length : P.vars.length;
  const toT = c => (red ? O.expand(P.Phi, c) : Array.from(c));
  const f = c => P.objOpt.rel(toT(c));
  const pen = 0.05;
  const F = c => { let s = 0; for (let j = 0; j < Dm; j++) s += (c[j] / P.delta) ** 2; return f(c) + pen * s / Dm; };
  return { Dm, toT, f, F };
}

module.exports = { SURFACES, GRID_OPT, GRID_VAL, setup, metrics, roughness, searchObjective, O };
