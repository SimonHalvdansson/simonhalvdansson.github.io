/* inr.js — in-browser INR training with TensorFlow.js (WebGPU, falling back to WebGL / CPU).
   The approach follows Every Frame an INR: Fourier features computed once (or per batch),
   a small MLP built from raw tf.variables, and full-batch Adam.                         */
(function () {
  const INR = { backend: null, ready: false, error: null };

  INR.init = function () {
    if (INR._init) return INR._init;
    INR._init = (async () => {
      if (!window.tf) throw new Error('TensorFlow.js did not load');
      const tryBackend = async name => {
        try {
          if (!tf.findBackendFactory(name)) return false;
          const ok = await tf.setBackend(name);
          if (!ok) return false;
          await tf.ready();
          return tf.getBackend() === name;
        } catch (e) { console.warn('backend', name, 'failed:', e); return false; }
      };
      if (navigator.gpu && await tryBackend('webgpu')) INR.backend = 'webgpu';
      else if (await tryBackend('webgl')) INR.backend = 'webgl';
      else { await tryBackend('cpu'); INR.backend = 'cpu'; }
      INR.ready = true;
      console.info('[INR] TensorFlow.js backend:', INR.backend);
      return INR.backend;
    })().catch(e => { INR.error = e; console.error(e); throw e; });
    return INR._init;
  };

  /* ---------------- model ---------------- */
  function randn(shape, scale, seed) {
    const g = U.gaussian(seed);
    const n = shape.reduce((a, b) => a * b, 1);
    const d = new Float32Array(n);
    for (let i = 0; i < n; i++) d[i] = g() * scale;
    return tf.tensor(d, shape);
  }

  /* tf.variable keeps its own reference to the data, so release the initial tensor */
  function makeVar(t) {
    const v = tf.variable(t, true);
    t.dispose();
    return v;
  }

  class Model {
    /* cfg: { inDim, outDim, features:'none'|'fourier', m, sigma, width, depth, outAct:'sigmoid'|'linear', seed, outBias } */
    constructor(cfg) {
      const c = this.cfg = Object.assign({ inDim: 2, outDim: 3, features: 'none', m: 128, sigma: 10, width: 128, depth: 3,
        outAct: 'sigmoid', seed: 7, outBias: 0 }, cfg);
      let d = c.inDim;
      this.B = null;
      if (c.features === 'fourier') {
        this.B = randn([c.inDim, c.m], c.sigma, c.seed + 999);
        d = 2 * c.m;
      }
      this.layers = [];
      this.vars = [];
      for (let l = 0; l < c.depth; l++) {
        const W = makeVar(randn([d, c.width], Math.sqrt(2 / d), c.seed + 101 + l));
        const b = makeVar(tf.zeros([c.width]));
        this.layers.push({ W, b, act: 'relu' });
        this.vars.push(W, b);
        d = c.width;
      }
      const Wo = makeVar(randn([d, c.outDim], Math.sqrt(1 / d), c.seed + 301));
      const bo = makeVar(tf.fill([c.outDim], c.outBias));
      this.layers.push({ W: Wo, b: bo, act: c.outAct });
      this.vars.push(Wo, bo);
    }
    get params() {
      return this.vars.reduce((a, v) => a + v.size, 0);
    }
    features(x) {
      if (!this.B) return x;
      return tf.tidy(() => {
        const p = tf.matMul(x, this.B).mul(2 * Math.PI);
        return tf.concat([tf.cos(p), tf.sin(p)], 1);
      });
    }
    mlp(h) {
      for (const L of this.layers) {
        h = tf.add(tf.matMul(h, L.W), L.b);
        if (L.act === 'relu') h = tf.relu(h);
        else if (L.act === 'sigmoid') h = tf.sigmoid(h);
      }
      return h;
    }
    forward(x) { return this.mlp(this.features(x)); }
    dispose() {
      this.vars.forEach(v => v.dispose());
      if (this.B) this.B.dispose();
      this.vars = []; this.layers = [];
    }
  }
  INR.Model = Model;

  /* ---------------- trainer base ---------------- */
  class Trainer {
    constructor(model, lr) {
      this.model = model;
      this.lr0 = lr;
      this.opt = tf.train.adam(lr);
      this.step = 0;
      this.hist = { steps: [], loss: [] };
      this.busy = false;
      this.stepsPerChunk = 4;
      this.disposed = false;
    }
    lossFn() { throw new Error('abstract'); }
    prepare() {}
    /* optional cosine decay: set this.lrEnd and this.decaySteps */
    schedule() {
      if (this.lrEnd == null || !this.decaySteps) return;
      const u = Math.min(1, this.step / this.decaySteps);
      this.setLR(this.lrEnd + 0.5 * (this.lr0 - this.lrEnd) * (1 + Math.cos(Math.PI * u)));
    }
    setLR(lr) { this.opt.learningRate = lr; }
    /* run n optimisation steps, read back the losses once at the end */
    async chunk(n = this.stepsPerChunk) {
      if (this.busy || this.disposed) return null;
      this.busy = true;
      const t0 = performance.now();
      const costs = [];
      try {
        for (let i = 0; i < n; i++) {
          this.prepare();
          const cost = this.opt.minimize(() => tf.tidy(() => this.lossFn()), true, this.model.vars);
          costs.push(cost);
          this.step++;
          this.schedule();
        }
        const st = tf.stack(costs);
        costs.forEach(c => c.dispose());
        const v = await st.data();
        st.dispose();
        for (let i = 0; i < v.length; i++) { this.hist.steps.push(this.step - v.length + 1 + i); this.hist.loss.push(v[i]); }
        this.lastMs = performance.now() - t0;
        this.lastN = n;
        return v[v.length - 1];
      } finally {
        this.busy = false;
        if (this.pendingDispose) this.dispose();
      }
    }
    /* adapt steps per chunk so one chunk takes roughly `targetMs` (smoothed per-step estimate;
       the first chunks include shader compilation, so they are only trusted downwards) */
    adapt(targetMs = 45, maxSteps = 64) {
      if (!this.lastMs || !this.lastN) return;
      const per = this.lastMs / this.lastN;
      this.perStep = this.perStep ? 0.75 * this.perStep + 0.25 * per : per;
      const want = Math.round(targetMs / this.perStep);
      this.stepsPerChunk = U.clamp(want, 1, maxSteps);
    }
    get lastLoss() { return this.hist.loss.length ? this.hist.loss[this.hist.loss.length - 1] : NaN; }
    /* safe to call while a chunk is in flight: the actual release waits for it */
    dispose() {
      if (this.disposed) return;
      if (this.busy) { this.pendingDispose = true; return; }
      this.disposed = true;
      this.free();
    }
    free() {
      if (this.opt) this.opt.dispose();
      this.model.dispose();
    }
  }

  /* ---------------- image fitting (full batch) ---------------- */
  class ImageFit extends Trainer {
    /* target: Float32Array (size*size*3) in [0,1] */
    constructor(model, target, size, lr) {
      super(model, lr);
      this.size = size;
      const N = size * size;
      const xy = new Float32Array(N * 2);
      for (let i = 0; i < N; i++) { xy[2 * i] = (i % size) / (size - 1); xy[2 * i + 1] = Math.floor(i / size) / (size - 1); }
      this.coords = tf.tensor2d(xy, [N, 2]);
      this.feats = model.features(this.coords);
      this.y = tf.tensor2d(target, [N, 3]);
    }
    lossFn() { return tf.mean(tf.squaredDifference(this.model.mlp(this.feats), this.y)); }
    /* swap the target (e.g. a new webcam frame); call between chunks */
    setTarget(target) {
      const old = this.y;
      this.y = tf.tensor2d(target, [this.size * this.size, 3]);
      old.dispose();
    }
    async predict() {
      const p = tf.tidy(() => this.model.mlp(this.feats));
      const d = await p.data();
      p.dispose();
      return d;
    }
    /* evaluate on any grid over [x0,x1] x [y0,y1] (the representation is continuous) */
    async render(size, x0 = 0, y0 = 0, x1 = 1, y1 = 1) {
      const N = size * size;
      const xy = new Float32Array(N * 2);
      for (let i = 0; i < N; i++) {
        xy[2 * i] = x0 + (x1 - x0) * (i % size) / (size - 1);
        xy[2 * i + 1] = y0 + (y1 - y0) * Math.floor(i / size) / (size - 1);
      }
      const p = tf.tidy(() => this.model.forward(tf.tensor2d(xy, [N, 2])));
      const d = await p.data();
      p.dispose();
      return d;
    }
    free() {
      super.free();
      if (this.feats !== this.coords) this.feats.dispose();
      this.coords.dispose();
      this.y.dispose();
    }
  }
  INR.ImageFit = ImageFit;

  /* ---------------- 1D signal fitting ---------------- */
  class SignalFit extends Trainer {
    constructor(model, t, y, lr) {
      super(model, lr);
      this.n = t.length;
      this.tt = tf.tensor2d(t, [t.length, 1]);
      this.feats = model.features(this.tt);
      this.y = tf.tensor2d(y, [y.length, 1]);
    }
    lossFn() { return tf.mean(tf.squaredDifference(this.model.mlp(this.feats), this.y)); }
    async predict() {
      const p = tf.tidy(() => this.model.mlp(this.feats));
      const d = await p.data();
      p.dispose();
      return d;
    }
    free() {
      super.free();
      if (this.feats !== this.tt) this.feats.dispose();
      this.tt.dispose(); this.y.dispose();
    }
  }
  INR.SignalFit = SignalFit;

  /* ---------------- CT: fit line integrals of f_theta to a sinogram ---------------- */
  class CTFit extends Trainer {
    /* opts: thetas (rad), ss (detector positions in [-1,1]), sino (Float32Array nang*ns),
             nray, nt, lr, lrEnd, steps (for schedule), seed                                    */
    constructor(model, opts) {
      super(model, opts.lr);
      Object.assign(this, { nray: 512, nt: 64, lrEnd: null, steps: 1000 }, opts);
      this.nang = this.thetas.length;
      this.ns = this.ss.length;
      this.rand = U.mulberry32(opts.seed || 3);
      this.cos = Float32Array.from(this.thetas, Math.cos);
      this.sin = Float32Array.from(this.thetas, Math.sin);
      this.pts = new Float32Array(this.nray * this.nt * 2);
      this.tgt = new Float32Array(this.nray);
      this.batch = null;
    }
    prepare() {
      const { nray, nt, rand, pts, tgt } = this;
      for (let r = 0; r < nray; r++) {
        const ia = Math.floor(rand() * this.nang), js = Math.floor(rand() * this.ns);
        const c = this.cos[ia], s = this.sin[ia], d = this.ss[js];
        tgt[r] = this.sino[ia * this.ns + js];
        for (let k = 0; k < nt; k++) {
          const t = ((k + rand()) / nt) * 2 - 1;           // stratified samples along the ray
          const o = (r * nt + k) * 2;
          pts[o] = d * c - t * s;
          pts[o + 1] = d * s + t * c;
        }
      }
      if (this.batch) { this.batch.p.dispose(); this.batch.y.dispose(); }
      this.batch = { p: tf.tensor2d(pts, [nray * nt, 2]), y: tf.tensor1d(tgt) };
    }
    lossFn() {
      const mu = this.model.forward(this.batch.p).reshape([this.nray, this.nt]);
      const pred = mu.sum(1).mul(2 / this.nt);
      return tf.mean(tf.squaredDifference(pred, this.batch.y));
    }
    schedule() {
      if (this.lrEnd === null) return;
      const s = this.step, S = this.steps, a = 0.6 * S;
      if (s > a) {
        const u = Math.min(1, (s - a) / (S - a));
        this.setLR(this.lrEnd + 0.5 * (this.lr0 - this.lrEnd) * (1 + Math.cos(Math.PI * u)));
      }
    }
    /* image of f_theta on an R x R grid over [-1,1]^2 (row 0 = top, y up) */
    async render(R) {
      if (!this.grid || this.gridR !== R) {
        if (this.grid) this.grid.dispose();
        const g = new Float32Array(R * R * 2);
        for (let i = 0; i < R; i++) for (let j = 0; j < R; j++) {
          g[(i * R + j) * 2] = -1 + 2 * j / (R - 1);
          g[(i * R + j) * 2 + 1] = 1 - 2 * i / (R - 1);
        }
        this.grid = tf.tensor2d(g, [R * R, 2]);
        this.gridR = R;
      }
      const p = tf.tidy(() => this.model.forward(this.grid));
      const d = await p.data();
      p.dispose();
      return d;
    }
    free() {
      super.free();
      if (this.batch) { this.batch.p.dispose(); this.batch.y.dispose(); }
      if (this.grid) this.grid.dispose();
    }
  }
  INR.CTFit = CTFit;

  /* ---------------- shader warm-up ----------------
     WebGPU compiles a pipeline per op and shape the first time it is used, which can take
     several seconds. Slides register small warm-up jobs using the exact shapes they train
     with; they run once, in the background, right after the backend is ready.          */
  const warmups = [];
  INR.addWarmup = fn => warmups.push(fn);
  INR.warmup = function () {
    if (INR._warm) return INR._warm;
    INR._warm = (async () => {
      await INR.init();
      if (INR.backend === 'cpu') return;
      const t0 = performance.now();
      for (const fn of warmups) {
        try { await fn(); } catch (e) { console.warn('[INR] warm-up failed', e); }
        await new Promise(r => setTimeout(r, 30));
      }
      console.info(`[INR] shader warm-up done in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    })();
    return INR._warm;
  };
  /* run a trainer for a couple of steps (+ one prediction) and release it */
  INR.warmTrainer = async (tr, predict = 'predict', arg) => {
    await tr.chunk(2);
    if (predict && tr[predict]) await tr[predict](arg);
    tr.dispose();
  };

  /* ---------------- a run loop bound to a slide context ----------------
     Trains `trainers` round-robin while ctx is alive. Training yields to the event loop with a
     MessageChannel (not requestAnimationFrame), so its speed does not depend on the display rate;
     onFrame(trainers) is called at most every `drawMs` (draw there). stopWhen(tr) ends a trainer. */
  const chan = new MessageChannel();
  const waiting = [];
  chan.port1.onmessage = () => { const r = waiting.shift(); if (r) r(); };
  const yieldMain = () => new Promise(r => { waiting.push(r); chan.port2.postMessage(0); });

  INR.run = function (ctx, trainers, { onFrame, stopWhen, targetMs = 40, maxSteps = 64, drawMs = 33, minRoundMs = 0 } = {}) {
    let stopped = false;
    const gen = ctx.gen;
    const alive = () => !stopped && ctx.gen === gen;
    (async () => {
      let lastDraw = 0;
      while (alive()) {
        const roundStart = performance.now();
        for (const tr of trainers) {
          if (!alive()) break;
          if (stopWhen && stopWhen(tr)) continue;
          await tr.chunk();
          tr.adapt(targetMs / trainers.length, maxSteps);
        }
        if (!alive()) break;
        const done = stopWhen && trainers.every(stopWhen);
        const now = performance.now();
        if (onFrame && (done || now - lastDraw >= drawMs)) { lastDraw = now; await onFrame(trainers); }
        if (done) break;
        const left = minRoundMs - (performance.now() - roundStart);
        if (left > 1) await new Promise(r => setTimeout(r, left));      // pacing, for demos meant to be watched
        else await yieldMain();
      }
    })().catch(e => console.error(e));
    return () => { stopped = true; };
  };

  window.INR = INR;
})();
