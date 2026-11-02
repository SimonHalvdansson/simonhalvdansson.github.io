/* deck.js — a small slide engine with build steps and per-slide animation contexts.

   Keys
     → ↓ PgDn Space Enter : next step / slide        ← ↑ PgUp Backspace : previous
     Home / End           : first / last slide       f : fullscreen
     r                    : restart current slide    b or . : blank screen
     o / Esc              : overview                 p : presenter view (notes + timer)
     digits + Enter       : jump to slide number

   Slides are <section class="slide" id="..."> elements inside #stage.
   Elements with data-step="k" appear at build step k; data-step-out="k" disappears at step k.
   Modules registered with Deck.register(id, {...}) get lifecycle callbacks:
     init(ctx)                 once, the first time the slide is shown
     enter(ctx, step)          when the slide becomes active (step = initial build step)
     step(ctx, step, prev)     when the build step changes while active
     leave(ctx)                when the slide stops being active (ctx is cancelled after)
     resize(ctx)               when the stage pixel ratio changes
     steps                     optional number of build steps (if not derivable from DOM)
*/
(function () {
  const W = 1920, H = 1080;
  const CANCEL = { cancelled: true };

  /* ---------------- animation contexts ---------------- */
  class Ctx {
    constructor(el, parent = null) {
      this.el = el;
      this.parent = parent;
      this.gen = 0;
      this.fns = new Set();
      this.children = new Set();
      if (parent) parent.children.add(this);
    }
    $(sel) { return this.el.querySelector(sel); }
    $$(sel) { return [...this.el.querySelectorAll(sel)]; }
    /* animate over `dur` ms; fn(easedT, rawT) every frame; resolves at the end */
    play(dur, fn, ease = U.Ease.inOutCubic) {
      const gen = this.gen;
      return new Promise((resolve, reject) => {
        const t0 = performance.now();
        const f = now => {
          if (gen !== this.gen) { this.fns.delete(f); reject(CANCEL); return; }
          const t = dur <= 0 ? 1 : Math.min(1, (now - t0) / dur);
          fn(ease(t), t);
          if (t >= 1) { this.fns.delete(f); resolve(); }
        };
        this.fns.add(f);
        f(t0);
      });
    }
    wait(ms) { return this.play(ms, () => {}, U.Ease.linear); }
    /* timer-based wait (independent of animation frames), still cancelled with the context */
    sleep(ms) {
      const gen = this.gen;
      return new Promise((resolve, reject) => setTimeout(() => (gen === this.gen ? resolve() : reject(CANCEL)), ms));
    }
    /* per-frame callback until cancelled (or until fn returns false). returns stop() */
    loop(fn) {
      const gen = this.gen;
      let last = performance.now();
      const f = now => {
        if (gen !== this.gen) { this.fns.delete(f); return; }
        now = Math.max(now, last);           // rAF timestamps can predate the loop's start
        const dt = Math.min(0.1, (now - last) / 1000); last = now;
        if (fn(now / 1000, dt) === false) this.fns.delete(f);
      };
      this.fns.add(f);
      return () => this.fns.delete(f);
    }
    /* run an async sequence; swallow cancellation */
    run(asyncFn) {
      return asyncFn(this).catch(e => { if (e !== CANCEL) console.error(e); });
    }
    child() { return new Ctx(this.el, this); }
    cancel() {
      this.gen++;
      const now = performance.now();
      for (const f of [...this.fns]) f(now);
      this.fns.clear();
      for (const c of this.children) c.cancel();
    }
    /* one failing animation must never stop the others (or the deck's frame loop) */
    tick(now) {
      for (const f of [...this.fns]) {
        try { f(now); } catch (e) { console.error(e); this.fns.delete(f); }
      }
      for (const c of this.children) c.tick(now);
    }
    get alive() { return true; }
  }

  /* ---------------- deck ---------------- */
  const Deck = {
    W, H, CANCEL,
    slides: [],
    modules: {},
    ctxs: [],
    inited: [],
    idx: 0,
    stepIdx: 0,
    scale: 1,
    jumpBuf: '',
    presenter: null,
    startTime: null,
    resizeFns: new Set(),

    register(id, mod) { this.modules[id] = mod; },
    pixelRatio() { return this.scale * (window.devicePixelRatio || 1); },
    onResize(fn) { this.resizeFns.add(fn); },

    init() {
      // ?static disables CSS transitions (used for screenshot testing)
      if (/static/.test(location.search)) document.body.classList.add('static');
      this.stage = document.getElementById('stage');
      this.slides = [...this.stage.querySelectorAll('section.slide')];
      this.ctxs = this.slides.map(el => new Ctx(el));
      this.inited = this.slides.map(() => false);

      // chrome
      this.progress = U.el('div', { id: 'progress' }, this.stage);
      this.slideno = U.el('div', { id: 'slideno' }, this.stage);
      this.blank = U.el('div', { id: 'blank' }, document.body);
      this.hint = U.el('div', { id: 'hint' }, document.body,
        '<b>f</b> fullscreen · <b>← →</b> navigate · <b>r</b> restart slide · <b>o</b> overview · <b>p</b> presenter view');
      this.jumpEl = U.el('div', { id: 'jump' }, document.body);
      this.buildOverview();

      // math
      U.typeset(this.stage);

      this.fit();
      window.addEventListener('resize', () => this.fit());
      document.addEventListener('keydown', e => this.onKey(e));
      document.addEventListener('fullscreenchange', () => setTimeout(() => this.fit(), 50));

      // initial position from hash
      const [s, st] = this.parseHash();
      this.goTo(s, st, true);

      // animation loop
      const tick = now => {
        requestAnimationFrame(tick);
        for (let i = 0; i < this.ctxs.length; i++) {
          if (this.ctxs[i].fns.size || this.ctxs[i].children.size) this.ctxs[i].tick(now);
        }
        this.updatePresenterClock();
      };
      requestAnimationFrame(tick);

      // show key hint briefly
      setTimeout(() => this.hint.classList.add('on'), 600);
      setTimeout(() => this.hint.classList.remove('on'), 5200);
    },

    fit() {
      const vw = window.innerWidth, vh = window.innerHeight;
      const s = Math.min(vw / W, vh / H);
      const changed = Math.abs(s - this.scale) > 1e-4;
      this.scale = s;
      this.stage.style.setProperty('--scale', s);
      if (changed) {
        for (const fn of this.resizeFns) fn();
        this.slides.forEach((el, i) => {
          const m = this.modules[el.id];
          if (this.inited[i] && m && m.resize) m.resize(this.ctxs[i]);
        });
      }
    },

    parseHash() {
      const m = location.hash.match(/^#\/?(\d+)(?:\.(\d+))?/);
      if (!m) return [0, 0];
      return [U.clamp(parseInt(m[1], 10) - 1, 0, this.slides.length - 1), m[2] ? parseInt(m[2], 10) : 0];
    },

    maxStep(i) {
      const el = this.slides[i];
      const m = this.modules[el.id];
      let n = 0;
      el.querySelectorAll('[data-step]').forEach(e => { n = Math.max(n, parseInt(e.dataset.step, 10) || 0); });
      el.querySelectorAll('[data-step-out]').forEach(e => { n = Math.max(n, parseInt(e.dataset.stepOut, 10) || 0); });
      if (m && typeof m.steps === 'number') n = Math.max(n, m.steps);
      return n;
    },

    applyStepClasses(i, step) {
      const el = this.slides[i];
      el.querySelectorAll('[data-step]').forEach(e => {
        e.classList.toggle('shown', (parseInt(e.dataset.step, 10) || 0) <= step);
      });
      el.querySelectorAll('[data-step-out]').forEach(e => {
        e.classList.toggle('gone', (parseInt(e.dataset.stepOut, 10) || 0) <= step);
      });
    },

    goTo(i, step = 0, initial = false) {
      i = U.clamp(i, 0, this.slides.length - 1);
      step = U.clamp(step, 0, this.maxStep(i));
      const prev = this.idx;
      if (!initial && i === prev) { this.setStep(step); return; }

      // leave previous
      if (!initial) {
        const pel = this.slides[prev];
        const pm = this.modules[pel.id];
        if (pm && pm.leave) { try { pm.leave(this.ctxs[prev]); } catch (e) { console.error(e); } }
        this.ctxs[prev].cancel();
        pel.classList.remove('active');
        pel.classList.add('leaving');
        setTimeout(() => pel.classList.remove('leaving'), 450);
      }

      this.idx = i;
      this.stepIdx = step;
      const el = this.slides[i];
      // keep the section kicker still when the previous slide had the same one
      const kNew = el.querySelector('.kicker'), kOld = this.slides[prev].querySelector('.kicker');
      el.classList.toggle('same-kicker', !initial && !!kNew && !!kOld && kNew.textContent === kOld.textContent);
      this.applyStepClasses(i, step);
      // disable transitions for elements when jumping into a slide so the built state appears at once
      el.classList.add('no-anim');
      el.classList.add('active');
      const m = this.modules[el.id];
      const ctx = this.ctxs[i];
      if (m && !this.inited[i]) {
        this.inited[i] = true;
        if (m.init) { try { m.init(ctx); } catch (e) { console.error(e); } }
      }
      if (m && m.enter) { try { m.enter(ctx, step); } catch (e) { console.error(e); } }
      requestAnimationFrame(() => el.classList.remove('no-anim'));

      if (this.startTime === null && !initial) this.startTime = performance.now();
      this.updateChrome();
    },

    setStep(step) {
      const i = this.idx;
      step = U.clamp(step, 0, this.maxStep(i));
      const prev = this.stepIdx;
      if (step === prev) return;
      this.stepIdx = step;
      this.applyStepClasses(i, step);
      const m = this.modules[this.slides[i].id];
      if (m && m.step) { try { m.step(this.ctxs[i], step, prev); } catch (e) { console.error(e); } }
      this.updateChrome();
    },

    next() {
      if (this.startTime === null) this.startTime = performance.now();
      if (this.stepIdx < this.maxStep(this.idx)) this.setStep(this.stepIdx + 1);
      else if (this.idx < this.slides.length - 1) this.goTo(this.idx + 1, 0);
    },
    prev() {
      if (this.stepIdx > 0) this.setStep(this.stepIdx - 1);
      else if (this.idx > 0) this.goTo(this.idx - 1, this.maxStep(this.idx - 1));
    },
    restart() {
      const i = this.idx;
      const el = this.slides[i];
      const m = this.modules[el.id];
      if (m && m.leave) { try { m.leave(this.ctxs[i]); } catch (e) { console.error(e); } }
      this.ctxs[i].cancel();
      if (m && m.restart) { try { m.restart(this.ctxs[i]); } catch (e) { console.error(e); } }
      this.stepIdx = 0;
      this.applyStepClasses(i, 0);
      if (m && m.enter) { try { m.enter(this.ctxs[i], 0); } catch (e) { console.error(e); } }
      this.updateChrome();
    },

    updateChrome() {
      const n = this.slides.length;
      this.progress.style.width = ((this.idx + 1) / n * 100) + '%';
      this.slideno.textContent = `${this.idx + 1} / ${n}`;
      this.slideno.style.display = this.slides[this.idx].dataset.chrome === 'off' ? 'none' : '';
      this.progress.style.display = this.slides[this.idx].dataset.chrome === 'off' ? 'none' : '';
      const h = `#/${this.idx + 1}` + (this.stepIdx ? `.${this.stepIdx}` : '');
      if (location.hash !== h) history.replaceState(null, '', h);
      document.querySelectorAll('#overview .ov').forEach((e, k) => e.classList.toggle('cur', k === this.idx));
      this.updatePresenter();
    },

    toggleFullscreen() {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
      else document.exitFullscreen();
    },

    buildOverview() {
      this.ov = U.el('div', { id: 'overview' }, document.body);
      this.slides.forEach((el, i) => {
        const t = el.dataset.title || (el.querySelector('h1,h2') || {}).textContent || el.id;
        const d = U.el('div', { class: 'ov' }, this.ov, `<small>${i + 1}</small>${t}`);
        d.addEventListener('click', () => { document.body.classList.remove('overview'); this.goTo(i, 0); });
      });
    },

    onKey(e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key;
      const mod = this.modules[this.slides[this.idx].id];
      if (mod && mod.keys && mod.keys.includes(k) && !this.jumpBuf) { mod.key(this.ctxs[this.idx], k, e); return; }
      if (/^[0-9]$/.test(k)) {
        this.jumpBuf += k; this.jumpEl.style.display = 'block'; this.jumpEl.textContent = '→ ' + this.jumpBuf;
        clearTimeout(this.jumpTimer);
        this.jumpTimer = setTimeout(() => { this.jumpBuf = ''; this.jumpEl.style.display = 'none'; }, 2500);
        return;
      }
      if (k === 'Enter' && this.jumpBuf) {
        const n = parseInt(this.jumpBuf, 10); this.jumpBuf = ''; this.jumpEl.style.display = 'none';
        document.body.classList.remove('overview');
        this.goTo(n - 1, 0); e.preventDefault(); return;
      }
      switch (k) {
        case 'ArrowRight': case 'ArrowDown': case 'PageDown': case ' ': case 'Enter':
          e.preventDefault(); this.next(); break;
        case 'ArrowLeft': case 'ArrowUp': case 'PageUp': case 'Backspace':
          e.preventDefault(); this.prev(); break;
        case 'Home': e.preventDefault(); this.goTo(0, 0); break;
        case 'End': e.preventDefault(); this.goTo(this.slides.length - 1, 0); break;
        case 'f': case 'F': this.toggleFullscreen(); break;
        case 'F5': e.preventDefault(); this.toggleFullscreen(); break;      // clicker "start presentation" button
        case 'r': case 'R': this.restart(); break;
        case 'b': case 'B': case '.': this.blank.classList.toggle('on'); break;
        case 'o': case 'O': case 'Escape':
          if (k === 'Escape' && !document.body.classList.contains('overview')) {
            if (document.fullscreenElement) return;
          }
          document.body.classList.toggle('overview'); break;
        case 'p': case 'P': this.openPresenter(); break;
        case 'h': case 'H': case '?': this.hint.classList.toggle('on'); break;
        default: {
          const m = this.modules[this.slides[this.idx].id];
          if (m && m.key) m.key(this.ctxs[this.idx], k, e);
        }
      }
    },

    /* ---------------- presenter view ---------------- */
    openPresenter() {
      if (this.presenter && !this.presenter.closed) { this.presenter.focus(); return; }
      const w = window.open('', 'inr-presenter', 'width=980,height=720');
      if (!w) {
        this.hint.textContent = 'The presenter window was blocked: allow pop-ups for this page and press P again.';
        this.hint.classList.add('on');
        setTimeout(() => this.hint.classList.remove('on'), 4000);
        return;
      }
      this.presenter = w;
      w.document.title = 'Presenter — INR talk';
      w.document.body.innerHTML = `
        <style>
          body{margin:0;font-family:"Fira Sans",system-ui,sans-serif;background:#16181f;color:#eceae4;display:flex;flex-direction:column;height:100vh}
          header{display:flex;justify-content:space-between;align-items:baseline;padding:18px 26px;border-bottom:1px solid #2b2e38}
          #pt{font-size:26px;font-weight:600} #pc{font:500 34px/1 ui-monospace,Menlo,monospace;color:#86b6ef}
          #pn{padding:22px 26px;font-size:24px;line-height:1.5;overflow:auto;flex:1;white-space:normal}
          #pn ul{margin:0;padding-left:1.1em} #pn li{margin:0 0 .45em}
          footer{padding:14px 26px;border-top:1px solid #2b2e38;color:#9a988f;font-size:18px;display:flex;justify-content:space-between}
          #ps{font:500 18px ui-monospace,Menlo,monospace}
        </style>
        <header><div id="pt"></div><div id="pc">00:00</div></header>
        <div id="pn"></div>
        <footer><div id="pnext"></div><div id="ps"></div></footer>`;
      w.document.addEventListener('keydown', e => this.onKey(e));
      this.updatePresenter();
    },
    updatePresenter() {
      const w = this.presenter;
      if (!w || w.closed) return;
      const el = this.slides[this.idx];
      const t = el.dataset.title || (el.querySelector('h1,h2') || {}).textContent || el.id;
      const notes = el.querySelector('aside.notes');
      const nx = this.slides[this.idx + 1];
      const nt = nx ? (nx.dataset.title || (nx.querySelector('h1,h2') || {}).textContent || nx.id) : '— end —';
      w.document.getElementById('pt').textContent = `${this.idx + 1}. ${t}`;
      w.document.getElementById('pn').innerHTML = notes ? notes.innerHTML : '<i>No notes.</i>';
      w.document.getElementById('pnext').textContent = 'Next: ' + nt;
      w.document.getElementById('ps').textContent = `step ${this.stepIdx} / ${this.maxStep(this.idx)}`;
    },
    updatePresenterClock() {
      const w = this.presenter;
      if (!w || w.closed) return;
      const s = this.startTime === null ? 0 : Math.floor((performance.now() - this.startTime) / 1000);
      const txt = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
      const c = w.document.getElementById('pc');
      if (c && c.textContent !== txt) c.textContent = txt;
    },
  };

  window.Deck = Deck;
})();
