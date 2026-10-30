// Run with: node tests/test_labs.js
"use strict";
const assert = require("node:assert/strict");
const M = require("../labs.js");
const near = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
const mu = [0, 0, 1], E = [[1, 0, 0], [0, 1, 0]];
const y = M.exp(mu, [0.2, 0.1, 0]);
M.log(mu, y).forEach((x, i) => near(x, [0.2, 0.1, 0][i]));
const P = [[0.025, 0], [0, 0.025]];
const low = [[0.003, 0], [0, 0.003]], high = [[0.09, 0], [0, 0.003]];
const a = M.update(mu, E, P, y, low), b = M.update(mu, E, P, y, high);
for (const state of [{mu, E, P}, b]) {
  const next = M.update(state.mu, state.E, state.P, y, low);
  const workspace = M.tangentWorkspace(state.mu, state.E, y, next.correction);
  for (const point of [...workspace.corners, workspace.observationTip, workspace.correctionTip]) {
    near(M.dot(state.mu, M.add(point, M.scale(state.mu, -1))), 0);
  }
  const flatStep = M.add(workspace.correctionTip, M.scale(state.mu, -1));
  M.exp(state.mu, flatStep).forEach((x, i) => near(x, next.mu[i]));
  const flatObservation = M.add(workspace.observationTip, M.scale(state.mu, -1));
  M.exp(state.mu, flatObservation).forEach((x, i) => near(x, y[i]));
  assert.ok(M.norm(workspace.observationTip) > 1);
}
assert.ok(b.correction[0] < a.correction[0]);
near(b.correction[1], a.correction[1]);
near(M.norm(b.mu), 1);
b.E.forEach(e => {near(M.dot(e, b.mu), 0); near(M.norm(e), 1);});
near(M.dot(b.E[0], b.E[1]), 0);
assert.ok(b.P[0][0] > 0 && b.P[1][1] > 0);
assert.ok(b.P[0][0] * b.P[1][1] - b.P[0][1] ** 2 > 0);
assert.ok(M.score(P, high, [0.2, 0.1]).distance2 < M.score(P, low, [0.2, 0.1]).distance2);
const angle = 0.6, O = [[Math.cos(angle), -Math.sin(angle)], [Math.sin(angle), Math.cos(angle)]];
const rotatedE = [[O[0][0], O[1][0], 0], [O[0][1], O[1][1], 0]];
const rotatedR = M.multiply(M.multiply(M.transpose(O), high), O);
const rotated = M.update(mu, rotatedE, P, y, rotatedR);
b.mu.forEach((x, i) => near(x, rotated.mu[i]));
for (const bearing of [0, 0.4, 1.2, Math.PI / 2]) {
  const theta = M.kentRadius(32, 10, bearing);
  near(32 * (1 - Math.cos(theta)) - 10 * Math.cos(2 * bearing) * Math.sin(theta) ** 2, 1.5);
}
assert.ok(M.kentRadius(32, 10, 0) > M.kentRadius(32, 10, Math.PI / 2));
// Observation noise lives at the observation, then moves into the belief frame.
const obsE = E.map(e => M.transport(mu, y, e));
const ellipse = M.rotatedCovariance(0.09, 0.01, Math.PI / 4);
near(ellipse[0][0], 0.05); near(ellipse[1][1], 0.05); near(ellipse[0][1], 0.04);
const aligned = M.observationCovariance(mu, E, y, obsE, ellipse);
ellipse.forEach((row, i) => row.forEach((x, j) => near(x, aligned[i][j])));
const principal = M.principalDirections(ellipse);
near(M.dot(principal[0], principal[1]), 0);
near(M.dot(principal[0], M.mv(ellipse, principal[0])), 0.09);
near(M.dot(principal[1], M.mv(ellipse, principal[1])), 0.01);
const confidenceLow = M.vmfUpdate(mu, 18, y, 4);
const confidenceHigh = M.vmfUpdate(mu, 18, y, 100);
assert.ok(M.norm(M.log(mu, confidenceHigh.mu)) > M.norm(M.log(mu, confidenceLow.mu)));
near(M.norm(confidenceHigh.mu), 1);
let accumulated = {mu, kappa:18};
for (let i = 0; i < 12; i++) accumulated = M.vmfUpdate(accumulated.mu, accumulated.kappa, y, 18);
assert.ok(accumulated.kappa > 18);
assert.ok(M.norm(M.log(accumulated.mu, y)) < M.norm(M.log(mu, y)));
let sequence = {mu, E, P};
for (let i = 0; i < 12; i++) {
  const R = M.observationCovariance(sequence.mu, sequence.E, y, obsE, ellipse);
  sequence = M.update(sequence.mu, sequence.E, sequence.P, y, R);
  near(M.norm(sequence.mu), 1);
  sequence.E.forEach(e => near(M.dot(e, sequence.mu), 0));
  assert.ok(sequence.P[0][0] * sequence.P[1][1] - sequence.P[0][1] ** 2 > 0);
}
console.log("Widget updates, repeated fusion, observation covariance alignment, directional weighting, and Kent contours passed.");
