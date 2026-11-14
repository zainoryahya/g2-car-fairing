// Line-delimited JSON evaluation server so that scipy's L-BFGS-B can optimise the same objective.
// stdin: {"cmd":"setup","sid":"S1","delta_mm":10,"space":"reduced"} | {"cmd":"F","c":[...]} | {"cmd":"metrics","c":[...]} | {"cmd":"field","t":[...]}
const PR = require('./problems.js');
const rl = require('readline').createInterface({ input: process.stdin });
let P = null, so = null;
rl.on('line', (line) => {
  const m = JSON.parse(line); let out;
  if (m.cmd === 'setup') { P = PR.setup(m.sid, m.delta_mm / 1000); so = PR.searchObjective(P, m.space); out = { Dm: so.Dm }; }
  else if (m.cmd === 'F') out = { F: so.F(m.c) };
  else if (m.cmd === 'field') { out = P.objVal.field(m.t); out.J0 = P.objVal.E0; }
  else if (m.cmd === 'metrics') { const t = so.toT(m.c); out = { ...PR.metrics(P, t), t: t.map(x => +x.toFixed(7)) }; }
  process.stdout.write(JSON.stringify(out) + '\n');
});
