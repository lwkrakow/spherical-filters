/* Qualitative teaching illustrations. Numerical helpers mirror the local
 * sphere conventions in research_examples.py; no data or latency is measured.
 */
"use strict";

const SphereMath = (() => {
  const add = (a, b) => a.map((x, i) => x + b[i]);
  const scale = (a, s) => a.map(x => x * s);
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const norm = a => Math.sqrt(dot(a, a));
  const unit = a => scale(a, 1 / norm(a));
  const tangent = (E, z) => add(scale(E[0], z[0]), scale(E[1], z[1]));
  const coordinates = (E, v) => E.map(e => dot(e, v));
  function exp(mu, v) {
    const r = norm(v);
    return unit(add(scale(mu, Math.cos(r)), scale(v, r < 1e-10 ? 1 : Math.sin(r) / r)));
  }
  function log(mu, y) {
    const c = Math.max(-1, Math.min(1, dot(mu, y)));
    const v = add(y, scale(mu, -c));
    const s = norm(v);
    if (c < 0 && s < 1e-8) throw new Error("Antipodal log map");
    return s < 1e-10 ? v : scale(v, Math.atan2(s, c) / s);
  }
  function transport(mu, next, v) {
    const denominator = 1 + dot(mu, next);
    if (denominator < 1e-8) throw new Error("Antipodal transport");
    return add(v, scale(add(mu, next), -dot(next, v) / denominator));
  }
  const sum2 = (A, B) => A.map((row, i) => row.map((x, j) => x + B[i][j]));
  const transpose = A => A[0].map((_, j) => A.map(row => row[j]));
  const multiply = (A, B) => A.map(row => transpose(B).map(col => dot(row, col)));
  const mv = (A, v) => A.map(row => dot(row, v));
  function rotatedCovariance(variance1, variance2, angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    return [[c * c * variance1 + s * s * variance2, c * s * (variance1 - variance2)],
      [c * s * (variance1 - variance2), s * s * variance1 + c * c * variance2]];
  }
  function principalDirections(C) {
    const angle = 0.5 * Math.atan2(2 * C[0][1], C[0][0] - C[1][1]);
    return [[Math.cos(angle), Math.sin(angle)], [-Math.sin(angle), Math.cos(angle)]];
  }
  function tangentWorkspace(mu, E, y, correction) {
    const residual = log(mu, y), step = tangent(E, correction);
    const z = coordinates(E, residual);
    const half = z.map((x, i) => Math.max(0.6, Math.abs(x) * 1.15, Math.abs(correction[i]) * 1.15));
    const corner = (a, b) => add(mu, tangent(E, [half[0] * a, half[1] * b]));
    return {mu, E, half, corners:[corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)],
      observationTip:add(mu, residual), correctionTip:add(mu, step)};
  }
  function observationCovariance(mu, E, y, observationE, R) {
    // Align the observation-centered noise ellipse with the prior's frame.
    // Parallel transport is a local approximation to measurement alignment.
    const moved = observationE.map(e => transport(y, mu, e));
    const A = E.map(e => moved.map(column => dot(e, column)));
    return multiply(multiply(A, R), transpose(A));
  }
  function vmfUpdate(mu, kappa, y, observationKappa) {
    const eta = add(scale(mu, kappa), scale(y, observationKappa));
    const nextKappa = norm(eta);
    return {mu:nextKappa > 1e-12 ? scale(eta, 1 / nextKappa) : mu.slice(), kappa:nextKappa};
  }
  function solve2(A, b) {
    // Cholesky solve for the positive-definite 2x2 tutorial covariances.
    const l00 = Math.sqrt(A[0][0]), l10 = A[1][0] / l00;
    const l11 = Math.sqrt(A[1][1] - l10 * l10);
    const w0 = b[0] / l00, w1 = (b[1] - l10 * w0) / l11;
    const x1 = w1 / l11;
    return [(w0 - l10 * x1) / l00, x1];
  }
  function update(mu, E, P, y, R) {
    const S = sum2(P, R), K = P.map(row => solve2(S, row));
    const z = coordinates(E, log(mu, y)), correction = mv(K, z);
    const next = exp(mu, tangent(E, correction));
    const nextE = E.map(e => transport(mu, next, e));
    const IminusK = K.map((row, i) => row.map((x, j) => (i === j ? 1 : 0) - x));
    const posterior = sum2(multiply(multiply(IminusK, P), transpose(IminusK)), multiply(multiply(K, R), transpose(K)));
    const r = norm(correction), sinc = r < 1e-10 ? 1 : Math.sin(r) / r;
    const a = r < 1e-10 ? [1, 0] : scale(correction, 1 / r);
    const J = a.map((x, i) => a.map((y, j) => (i === j ? sinc : 0) + (1 - sinc) * x * y));
    const nextP = multiply(multiply(J, posterior), transpose(J));
    nextP[0][1] = nextP[1][0] = (nextP[0][1] + nextP[1][0]) / 2;
    return {mu:next, E:nextE, P:nextP, K, correction};
  }
  function score(P, R, z) {
    const S = sum2(P, R), distance2 = dot(z, solve2(S, z));
    const determinant = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    return {distance2, nll:0.5 * (distance2 + Math.log(determinant))};
  }
  function kentRadius(kappa, beta, bearing, drop = 1.5) {
    // Exact equal-density contour, NOT a probability/credible contour.
    let lo = 0, hi = Math.PI;
    for (let i = 0; i < 45; i++) {
      const theta = (lo + hi) / 2;
      const loss = kappa * (1 - Math.cos(theta)) - beta * Math.cos(2 * bearing) * Math.sin(theta) ** 2;
      if (loss < drop) lo = theta; else hi = theta;
    }
    return (lo + hi) / 2;
  }
  return {add, scale, dot, norm, unit, tangent, coordinates, exp, log, transport,
    sum2, multiply, transpose, mv, rotatedCovariance, observationCovariance,
    vmfUpdate, principalDirections, tangentWorkspace, solve2, update, score, kentRadius};
})();

if (typeof module !== "undefined") module.exports = SphereMath;

if (typeof document !== "undefined") {
  const M = SphereMath;
  const COLORS = {blue:"#78dcf5", green:"#83e2b0", amber:"#ffce85", red:"#ff939b", pink:"#f4a7f1"};
  const ORIGIN = [0, 0, 1], BASIS = [[1, 0, 0], [0, 1, 0]];
  const CHI2_95_2D = 5.991464547;
  const byId = id => document.getElementById(id);

  class SpherePlot {
    constructor(canvas) {
      this.canvas = canvas; this.ctx = canvas.getContext("2d");
      this.yaw = -0.38; this.pitch = 0.42; this.sceneScale = 126; this.onredraw = () => {};
      canvas.classList.add("rotatable"); canvas.tabIndex = 0;
      canvas.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight ArrowUp ArrowDown Home");
      const reset = () => {this.yaw = -0.38; this.pitch = 0.42; this.onredraw();};
      let drag = null;
      canvas.addEventListener("pointerdown", event => {
        if (event.button !== 0 || drag) return;
        drag = {id:event.pointerId, x:event.clientX, y:event.clientY};
        canvas.setPointerCapture(event.pointerId);
        canvas.classList.add("dragging"); canvas.focus({preventScroll:true});
      });
      canvas.addEventListener("pointermove", event => {
        if (!drag || event.pointerId !== drag.id) return;
        this.yaw += (event.clientX - drag.x) * 0.008;
        this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + (event.clientY - drag.y) * 0.008));
        drag.x = event.clientX; drag.y = event.clientY; this.onredraw();
      });
      const release = () => {drag = null; canvas.classList.remove("dragging");};
      for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) canvas.addEventListener(event, release);
      canvas.addEventListener("dblclick", reset);
      canvas.addEventListener("keydown", event => {
        if (event.key === "Home") {event.preventDefault(); reset(); return;}
        const directions = {ArrowLeft:[-0.12, 0], ArrowRight:[0.12, 0], ArrowUp:[0, -0.12], ArrowDown:[0, 0.12]};
        if (!directions[event.key]) return;
        event.preventDefault(); this.yaw += directions[event.key][0];
        this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + directions[event.key][1]));
        this.onredraw();
      });
    }
    rotate(v) {
      const yaw = this.yaw, pitch = this.pitch;
      const x = v[0] * Math.cos(yaw) + v[2] * Math.sin(yaw);
      const z = -v[0] * Math.sin(yaw) + v[2] * Math.cos(yaw);
      const y = v[1] * Math.cos(pitch) - z * Math.sin(pitch);
      return [x, y, v[1] * Math.sin(pitch) + z * Math.cos(pitch)];
    }
    project(v) {
      const [x, y, z] = this.rotate(v);
      return [this.canvas.width / 2 + x * this.sceneScale,
        this.canvas.height / 2 - y * this.sceneScale, z];
    }
    fitScene(points) {
      const rotated = points.map(p => this.rotate(p));
      const extentX = Math.max(1, ...rotated.map(p => Math.abs(p[0])));
      const extentY = Math.max(1, ...rotated.map(p => Math.abs(p[1])));
      this.sceneScale = Math.min(126, (this.canvas.width / 2 - 45) / extentX,
        (this.canvas.height / 2 - 42) / extentY);
    }
    clear() {
      const ctx = this.ctx, w = this.canvas.width, h = this.canvas.height, radius = this.sceneScale;
      ctx.clearRect(0, 0, w, h);
      const gradient = ctx.createRadialGradient(w / 2 - radius * 0.38, h / 2 - radius * 0.4, 4, w / 2, h / 2, radius);
      gradient.addColorStop(0, "#253c56"); gradient.addColorStop(1, "#0b1422");
      ctx.fillStyle = gradient; ctx.strokeStyle = "#344a64"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(w / 2, h / 2, radius, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      for (const plane of [0, 1]) {
        const points = Array.from({length:121}, (_, i) => {
          const a = i * 2 * Math.PI / 120;
          return plane === 0 ? [Math.cos(a), Math.sin(a), 0] : [0, Math.cos(a), Math.sin(a)];
        });
        this.path(points, "#31475e", false, 1);
      }
    }
    path(points, color, dashed = false, width = 2) {
      const ctx = this.ctx;
      ctx.strokeStyle = color; ctx.lineWidth = width;
      const projected = points.map(v => this.project(v));
      for (const front of [false, true]) {
        ctx.globalAlpha = front ? 1 : 0.24;
        ctx.setLineDash(!front || dashed ? [5, 5] : []);
        ctx.beginPath();
        let connected = false;
        for (const p of projected) {
          if ((p[2] >= 0) !== front) {connected = false; continue;}
          if (!connected) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
          connected = true;
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1; ctx.setLineDash([]);
    }
    dot(v, color, label = "", dx = 10, dy = -10, radius = 5) {
      const p = this.project(v), ctx = this.ctx;
      ctx.globalAlpha = p[2] >= 0 ? 1 : 0.35;
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(p[0], p[1], radius, 0, Math.PI * 2); ctx.fill();
      if (label) {
        ctx.font = "bold 14px system-ui"; ctx.fillStyle = color;
        ctx.fillText(label, p[0] + dx, p[1] + dy);
      }
      ctx.globalAlpha = 1;
    }
    axes(mu, E) {
      E.forEach((e, i) => {
        const points = Array.from({length:31}, (_, j) => M.exp(mu, M.scale(e, (j / 30 - 0.5) * 1.5)));
        this.path(points, i === 0 ? COLORS.red : COLORS.blue, false, 1.2);
      });
    }
    flatLine(a, b, color, dashed = false, width = 1) {
      const ctx = this.ctx, p = this.project(a), q = this.project(b);
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dashed ? [4, 4] : []);
      ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
      ctx.setLineDash([]);
    }
    plane(workspace) {
      const ctx = this.ctx, corners = workspace.corners.map(p => this.project(p));
      ctx.fillStyle = "rgba(120, 220, 245, 0.10)";
      ctx.strokeStyle = "rgba(120, 220, 245, 0.70)"; ctx.lineWidth = 1.2;
      ctx.beginPath(); corners.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
      ctx.closePath(); ctx.fill(); ctx.stroke();
      for (const axis of [0, 1]) for (const fraction of [-0.5, 0, 0.5]) {
        const coordinates = [-1, 1].map(end => {
          const z = axis === 0 ? [fraction * workspace.half[0], end * workspace.half[1]]
            : [end * workspace.half[0], fraction * workspace.half[1]];
          return M.add(workspace.mu, M.tangent(workspace.E, z));
        });
        this.flatLine(...coordinates, "rgba(120, 220, 245, 0.22)");
      }
      ctx.font = "13px system-ui"; ctx.fillStyle = COLORS.blue;
      ctx.fillText("Tangent plane at the current belief", 14, this.canvas.height - 16);
    }
    flatArrow(a, b, color) {
      const ctx = this.ctx, p = this.project(a), q = this.project(b);
      const dx = q[0] - p[0], dy = q[1] - p[1], length = Math.hypot(dx, dy);
      if (length < 1) return;
      this.flatLine(a, b, color, false, 2.5);
      const angle = Math.atan2(dy, dx), head = Math.min(8, length * 0.4);
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.moveTo(q[0], q[1]);
      ctx.lineTo(q[0] - head * Math.cos(angle - 0.45), q[1] - head * Math.sin(angle - 0.45));
      ctx.lineTo(q[0] - head * Math.cos(angle + 0.45), q[1] - head * Math.sin(angle + 0.45));
      ctx.closePath(); ctx.fill();
    }
    workspaceVectors(workspace, observation, preview) {
      this.flatLine(observation, workspace.observationTip, COLORS.amber, true);
      this.flatLine(workspace.correctionTip, preview, COLORS.green, true);
      this.flatArrow(workspace.mu, workspace.observationTip, COLORS.amber);
      this.flatArrow(workspace.mu, workspace.correctionTip, COLORS.green);
    }
    covariance(mu, E, C, color, dashed = false, contourScale = CHI2_95_2D) {
      const l00 = Math.sqrt(C[0][0]), l10 = C[1][0] / l00;
      const l11 = Math.sqrt(Math.max(0, C[1][1] - l10 * l10));
      const points = Array.from({length:181}, (_, i) => {
        const a = i * 2 * Math.PI / 180;
        const z = M.scale([l00 * Math.cos(a), l10 * Math.cos(a) + l11 * Math.sin(a)], Math.sqrt(contourScale));
        return M.exp(mu, M.tangent(E, z));
      });
      this.path(points, color, dashed);
    }
    vmf(mu, kappa, color, dashed = false) {
      const cosine = kappa > 1e-10 ? 1 + Math.log(0.05 + 0.95 * Math.exp(-2 * kappa)) / kappa : -0.9;
      const radius = Math.acos(Math.max(-1, Math.min(1, cosine)));
      const E = BASIS.map(e => M.transport(ORIGIN, mu, e));
      const points = Array.from({length:181}, (_, i) => {
        const a = i * 2 * Math.PI / 180;
        return M.exp(mu, M.tangent(E, [radius * Math.cos(a), radius * Math.sin(a)]));
      });
      this.path(points, color, dashed);
    }
  }

  const radians = degrees => degrees * Math.PI / 180;
  const degrees = angle => angle * 180 / Math.PI;
  const value = id => Number(byId(id).value);
  function label(id, suffix = "°") {byId(`${id}-value`).textContent = `= ${value(id)}${suffix}`;}
  function observation(prefix) {
    const bearing = radians(value(`${prefix}-bearing`)), distance = radians(value(`${prefix}-distance`));
    const y = M.exp(ORIGIN, [distance * Math.cos(bearing), distance * Math.sin(bearing), 0]);
    return {y, E:BASIS.map(e => M.transport(ORIGIN, y, e))};
  }
  function resetControls(prefix) {
    document.querySelectorAll(`input[id^="${prefix}-"]`).forEach(input => {
      if (input.type === "checkbox") input.checked = input.defaultChecked;
      else input.value = input.defaultValue;
    });
  }
  function presets(prefix, options, draw) {
    document.querySelectorAll(`[data-preset^="${prefix}:"]`).forEach(button => {
      button.addEventListener("click", () => {
        const setting = options[button.dataset.preset.split(":")[1]];
        for (const [name, number] of Object.entries(setting)) byId(`${prefix}-${name}`).value = number;
        draw();
      });
    });
  }
  function correctionPath(plot, mu, preview) {
    const correction = M.log(mu, preview);
    plot.path(Array.from({length:31}, (_, i) => M.exp(mu, M.scale(correction, i / 30))), COLORS.green, false, 3);
  }
  function fusionDots(plot, state, obs, preview) {
    plot.dot(ORIGIN, "#92a3b8");
    correctionPath(plot, state.mu, preview.mu);
    plot.dot(state.mu, COLORS.blue, "belief", -65, 22);
    plot.dot(obs.y, COLORS.amber, "observation", 12, 7);
    plot.dot(preview.mu, COLORS.green, "preview", -70, -16);
  }

  if (byId("vmf-canvas")) {
    const plot = new SpherePlot(byId("vmf-canvas"));
    let state = {mu:ORIGIN.slice(), kappa:18}, count = 0;
    const preview = () => {
      const obs = observation("vmf");
      return {obs, next:M.vmfUpdate(state.mu, state.kappa, obs.y, value("vmf-kappa"))};
    };
    const draw = () => {
      const {obs, next} = preview(), k = value("vmf-kappa");
      plot.clear();
      plot.vmf(state.mu, state.kappa, COLORS.blue);
      plot.vmf(obs.y, k, COLORS.amber, true);
      plot.vmf(next.mu, next.kappa, COLORS.green);
      fusionDots(plot, state, obs, next);
      label("vmf-distance"); label("vmf-bearing"); label("vmf-kappa", "");
      const confidence = k > state.kappa * 1.1 ? "more confident than" : k < state.kappa * 0.9 ? "less confident than" : "about as confident as";
      byId("vmf-readout").textContent = `${count} observation${count === 1 ? "" : "s"} applied. Preview moves ${degrees(M.norm(M.log(state.mu, next.mu))).toFixed(1)}°. This observation is ${confidence} the current belief. Its noise footprint stays circular.`;
    };
    for (const id of ["vmf-distance", "vmf-bearing", "vmf-kappa"]) byId(id).addEventListener("input", draw);
    byId("vmf-update").addEventListener("click", () => {state = preview().next; count++; draw();});
    byId("vmf-reset").addEventListener("click", () => {state = {mu:ORIGIN.slice(), kappa:18}; count = 0; resetControls("vmf"); draw();});
    presets("vmf", {precise:{kappa:100}, noisy:{kappa:4}}, draw);
    plot.onredraw = draw; draw();
  }

  function gaussianLab(prefix, kent = false) {
    if (!byId(`${prefix}-canvas`)) return;
    const plot = new SpherePlot(byId(`${prefix}-canvas`));
    const initialState = () => ({mu:ORIGIN.slice(), E:BASIS.map(e => e.slice()), P:[[0.04, 0], [0, 0.04]]});
    let state = initialState(), count = 0, notice = "";
    const preview = () => {
      const obs = observation(prefix);
      if (kent) {
        obs.kappa = 1 / radians(value("kent-noise")) ** 2;
        obs.beta = obs.kappa * value("kent-shape") / 200;
        obs.tilt = radians(value("kent-tilt"));
        obs.R = M.rotatedCovariance(1 / (obs.kappa - 2 * obs.beta), 1 / (obs.kappa + 2 * obs.beta), obs.tilt);
      } else {
        obs.R = M.rotatedCovariance(radians(value("filter-noise1")) ** 2, radians(value("filter-noise2")) ** 2, radians(value("filter-tilt")));
      }
      const R = M.observationCovariance(state.mu, state.E, obs.y, obs.E, obs.R);
      return {obs, R, next:M.update(state.mu, state.E, state.P, obs.y, R)};
    };
    const draw = () => {
      const {obs, R, next} = preview();
      const workspace = !kent && byId("filter-plane").checked
        ? M.tangentWorkspace(state.mu, state.E, obs.y, next.correction) : null;
      if (workspace) plot.fitScene([...workspace.corners, obs.y, next.mu]);
      else plot.sceneScale = 126;
      plot.clear();
      if (workspace) plot.plane(workspace);
      plot.axes(state.mu, state.E);
      const contour = kent ? 3 : CHI2_95_2D;
      plot.covariance(state.mu, state.E, state.P, COLORS.blue, false, contour);
      if (kent) {
        const points = Array.from({length:181}, (_, i) => {
          const a = i * 2 * Math.PI / 180, r = M.kentRadius(obs.kappa, obs.beta, a);
          return M.exp(obs.y, M.tangent(obs.E, [r * Math.cos(a + obs.tilt), r * Math.sin(a + obs.tilt)]));
        });
        plot.path(points, COLORS.amber, true);
        label("kent-noise"); label("kent-shape", "%"); label("kent-tilt");
      } else {
        plot.covariance(obs.y, obs.E, obs.R, COLORS.amber, true);
        for (const id of ["filter-noise1", "filter-noise2", "filter-tilt"]) label(id);
        byId("filter-covariance").textContent = R.map(row => `[ ${row.map(x => x.toFixed(5).padStart(8)).join("  ")} ]`).join("\n");
      }
      plot.covariance(next.mu, next.E, next.P, COLORS.green, false, contour);
      if (workspace) plot.workspaceVectors(workspace, obs.y, next.mu);
      fusionDots(plot, state, obs, next);
      label(`${prefix}-distance`); label(`${prefix}-bearing`);
      byId(`${prefix}-readout`).textContent = `${count} observation${count === 1 ? "" : "s"} applied. Preview moves ${degrees(M.norm(M.log(state.mu, next.mu))).toFixed(1)}°. ${notice || "Change the noise footprint to see how its size and direction affect the update."}`;
    };
    document.querySelectorAll(`input[id^="${prefix}-"]`).forEach(input => input.addEventListener("input", () => {notice = ""; draw();}));
    byId(`${prefix}-update`).addEventListener("click", () => {state = preview().next; count++; notice = "The preview is the next possible update; blue now holds the belief you just accepted."; draw();});
    byId(`${prefix}-reset`).addEventListener("click", () => {state = initialState(); count = 0; notice = ""; resetControls(prefix); draw();});
    if (!kent) byId("filter-predict").addEventListener("click", () => {
      state.P = M.sum2(state.P, [[0.015, 0], [0, 0.015]]);
      notice = "Added uncertainty to the belief; the observation can now have more influence.";
      draw();
    });
    presets(prefix, kent ? {
      round:{shape:0, noise:8, tilt:0},
      sideways:{shape:80, noise:8, tilt:0},
      tilted:{shape:80, noise:8, tilt:45},
    } : {
      precise:{noise1:3, noise2:3, tilt:0},
      sideways:{noise1:18, noise2:4, tilt:0},
      upright:{noise1:18, noise2:4, tilt:90},
      tilted:{noise1:18, noise2:4, tilt:45},
    }, () => {notice = ""; draw();});
    plot.onredraw = draw; draw();
  }
  gaussianLab("kent", true);
  gaussianLab("filter");

  if (byId("comparison-canvas")) {
    const plot = new SpherePlot(byId("comparison-canvas"));
    // Deliberately overlapping physical sensitivities. These are illustrated
    // outputs of a calibrated pose model, not results of a PLS fit.
    const W = [[1, 0.35], [0.2, 0.9]], floor = 0.0016;
    const draw = () => {
      const supervised = byId("comparison-mode").value === "tpls";
      const correlation = byId("comparison-coupled").checked ? 0.75 : 0;
      const samples = Array.from({length:40}, (_, i) => {
        const r = Math.sqrt(-2 * Math.log((i + 0.5) / 40));
        const a = i * 2.399963229728653;
        const x = r * Math.cos(a), y = r * Math.sin(a);
        return M.mv(W, [0.16 * x, 0.12 * (correlation * x + Math.sqrt(1 - correlation ** 2) * y)]);
      });
      const mean = samples.reduce((sum, z) => M.add(sum, M.scale(z, 1 / samples.length)), [0, 0]);
      const centered = samples.map(z => M.add(z, M.scale(mean, -1)));
      const training = [[floor, 0], [0, floor]];
      for (const z of centered) for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) training[i][j] += z[i] * z[j] / (centered.length - 1);
      const pose = [[radians(value("comparison-azimuth")) ** 2, 0], [0, radians(value("comparison-elevation")) ** 2]];
      const conditional = M.sum2(M.multiply(M.multiply(W, pose), M.transpose(W)), [[floor, 0], [0, floor]]);
      plot.clear();
      for (const z of centered) plot.dot(M.exp(ORIGIN, M.tangent(BASIS, z)), "#718399", "", 0, 0, 2.5);
      plot.covariance(ORIGIN, BASIS, supervised ? conditional : training, COLORS.green);
      const arrows = supervised ? M.transpose(W) : M.principalDirections(training);
      arrows.forEach((z, i) => {
        const direction = M.unit(M.tangent(BASIS, z));
        const end = M.exp(ORIGIN, M.scale(direction, 0.62));
        const color = supervised ? (i === 0 ? COLORS.red : COLORS.blue) : (i === 0 ? COLORS.amber : COLORS.pink);
        plot.path(Array.from({length:31}, (_, j) => M.exp(ORIGIN, M.scale(direction, 0.62 * j / 30))), color, false, 2);
        plot.dot(end, color, supervised ? (i === 0 ? "azimuth" : "elevation") : `PC ${i + 1}`, 7, -7, 3);
      });
      plot.dot(ORIGIN, COLORS.blue, "reference", -75, 20);
      label("comparison-azimuth"); label("comparison-elevation");
      byId("comparison-azimuth").disabled = !supervised;
      byId("comparison-elevation").disabled = !supervised;
      byId("comparison-readout").textContent = supervised
        ? "The pose-related model has named azimuth and elevation sensitivities. Change their uncertainty separately: the ellipse follows their combined effects, even where the feature directions overlap."
        : "WC-TPCA follows the cloud's longest and shortest directions. Toggle the orbit-and-climb pattern to see those axes turn. Connecting these axes to physical angle changes takes an additional mapping.";
    };
    for (const id of ["comparison-mode", "comparison-coupled", "comparison-azimuth", "comparison-elevation"]) {
      byId(id).addEventListener("input", draw); byId(id).addEventListener("change", draw);
    }
    plot.onredraw = draw; draw();
  }

  if (byId("association-canvas")) {
    const plot = new SpherePlot(byId("association-canvas"));
    const P = [[0.004, 0], [0, 0.004]];
    const candidates = [{name:"A", z:[0.30, 0]}, {name:"B", z:[0, 0.30]}];
    const draw = () => {
      const directional = byId("association-mode").value === "directional";
      const ratio = Number(byId("association-ratio").value);
      const R = [[0.003 * (directional ? ratio : 1), 0], [0, 0.003]];
      plot.clear(); plot.axes(ORIGIN, BASIS);
      plot.covariance(ORIGIN, BASIS, M.sum2(P, R), COLORS.green);
      plot.dot(ORIGIN, COLORS.blue, "stored", -65, 20);
      const descriptions = [];
      for (const c of candidates) {
        plot.dot(M.exp(ORIGIN, M.tangent(BASIS, c.z)), COLORS.amber, c.name);
        const s = M.score(P, R, c.z);
        descriptions.push(`${c.name}: ${s.distance2 <= CHI2_95_2D ? "remains a plausible candidate" : "outside the illustrative gate"}.`);
      }
      if (byId("association-distractor").checked) {
        const z = [0.27, -0.04];
        plot.dot(M.exp(ORIGIN, M.tangent(BASIS, z)), COLORS.pink, "D", 10, 22);
        descriptions.push(M.score(P, R, z).distance2 <= CHI2_95_2D
          ? "D also passes: tolerance alone does not establish identity."
          : "D is outside this gate. Increasing tolerance can also admit distractors.");
      }
      byId("association-ratio-value").textContent = `= ${ratio}×`;
      byId("association-ratio").disabled = !directional;
      byId("association-readout").textContent = descriptions.join(" ");
    };
    for (const id of ["association-mode", "association-ratio", "association-distractor"]) {
      byId(id).addEventListener("input", draw);
      byId(id).addEventListener("change", draw);
    }
    plot.onredraw = draw; draw();
  }
}
