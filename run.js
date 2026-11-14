#!/usr/bin/env node
// Fairing experiment runner (metaheuristics + thin-plate baseline).
// Usage: node run.js [--runs 30] [--budget 6000] [--deltas 5,10,20] [--surfaces S1,S2,S3,S4]
//                    [--algos jde,pso,gwo] [--spaces full,reduced] [--out results] [--threads N]
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const fs = require('fs'), path = require('path'), os = require('os');
const PR = require('./src/problems.js'), O = PR.O;

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith('--') ? a.concat([[x.slice(2), arr[i + 1]]]) : a), []));
const CFG = {
  runs: +(args.runs || 30), budget: +(args.budget || 6000),
  deltas: (args.deltas || '5,10,20').split(',').map(Number),
  surfaces: (args.surfaces || 'S1,S2,S3,S4').split(','),
  algos: (args.algos || 'jde,pso,gwo').split(','), spaces: (args.spaces || 'full,reduced').split(','),
  out: args.out || 'results', threads: +(args.threads || os.cpus().length), checkpoints: 50, p: 3
};

function curve(hist, budget, K) {
  const out = []; let h = 0, cur = hist[0][1];
  for (let k = 0; k <= K; k++) { const e = budget * k / K; while (h < hist.length && hist[h][0] <= e) { cur = hist[h][1]; h++; } out.push(+cur.toFixed(6)); }
  return out;
}
const cache = new Map();
function problem(sid, dmm) { const k = sid + '/' + dmm; if (!cache.has(k)) cache.set(k, PR.setup(sid, dmm / 1000, CFG.p)); return cache.get(k); }

function runTask(task) {
  const { sid, dmm, algo, space, seed } = task, P = problem(sid, dmm), delta = dmm / 1000;
  const so = PR.searchObjective(P, space);
  const seeds = [new Float64Array(so.Dm)];
  if (algo === 'hyb') seeds.push(Float64Array.from(space === 'reduced' ? O.project(P.Phi, P.tp, delta) : P.tp));
  const t0 = process.hrtime.bigint();
  const r = O.runMeta(algo === 'hyb' ? 'jde' : algo, so.f, so.Dm, delta, CFG.budget, seeds, null, seed);
  const sec = Number(process.hrtime.bigint() - t0) / 1e9;
  const t = so.toT(r.best);
  return { sid, delta_mm: dmm, algo, space, seed, D: so.Dm, evals: r.evals, sec: +sec.toFixed(3), F_pen: r.bestF,
           ...PR.metrics(P, t), curve: curve(r.hist, CFG.budget, CFG.checkpoints), t: t.map(x => +x.toFixed(7)) };
}

if (isMainThread) {
  fs.mkdirSync(CFG.out, { recursive: true });
  const runsFile = path.join(CFG.out, 'runs.jsonl');
  fs.writeFileSync(runsFile, '');
  const info = {};
  for (const sid of CFG.surfaces) for (const dmm of CFG.deltas) {
    const P = problem(sid, dmm);
    info[sid] = { ...PR.SURFACES[sid], ctrlNet: [P.net.length, P.net[0].length], freeVars: P.vars.length, reducedVars: P.Phi[0].length };
    fs.appendFileSync(runsFile, JSON.stringify({ sid, delta_mm: dmm, algo: 'tp', space: 'full', seed: 0, D: P.vars.length, evals: 1, sec: 0, F_pen: null, ...PR.metrics(P, P.tp), curve: null, t: P.tp.map(x => +x.toFixed(7)) }) + '\n');
  }
  fs.writeFileSync(path.join(CFG.out, 'config.json'), JSON.stringify({ ...CFG, gridOpt: PR.GRID_OPT, gridVal: PR.GRID_VAL, surfaces_info: info, node: process.version, date: new Date().toISOString() }, null, 2));
  const tasks = [];
  for (const sid of CFG.surfaces) for (const dmm of CFG.deltas) for (const algo of CFG.algos) for (const space of CFG.spaces) for (let s = 1; s <= CFG.runs; s++) tasks.push({ sid, dmm, algo, space, seed: s });
  let next = 0, done = 0; const T0 = Date.now();
  console.log(`${tasks.length} runs on ${CFG.threads} threads; budget ${CFG.budget}; deltas ${CFG.deltas} mm; surfaces ${CFG.surfaces}`);
  for (let w = 0; w < CFG.threads; w++) {
    const wk = new Worker(__filename, { workerData: { cfg: CFG } });
    const feed = () => { if (next < tasks.length) wk.postMessage(tasks[next++]); else wk.terminate(); };
    wk.on('message', (res) => {
      fs.appendFileSync(runsFile, JSON.stringify(res) + '\n'); done++;
      if (done % 20 === 0 || done === tasks.length) console.log(`${done}/${tasks.length}  (${((Date.now() - T0) / 1000).toFixed(0)} s)`);
      feed();
    });
    wk.on('online', feed);
  }
} else {
  Object.assign(CFG, workerData.cfg);
  parentPort.on('message', (task) => parentPort.postMessage(runTask(task)));
}
