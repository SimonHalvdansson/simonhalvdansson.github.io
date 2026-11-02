/* util.js — small shared helpers (easing, canvas, svg, rng, formatting) */
(function () {
  const U = {};

  U.Ease = {
    linear: t => t,
    inQuad: t => t * t,
    outQuad: t => 1 - (1 - t) * (1 - t),
    inOutQuad: t => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2),
    inCubic: t => t * t * t,
    outCubic: t => 1 - Math.pow(1 - t, 3),
    inOutCubic: t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
    outQuint: t => 1 - Math.pow(1 - t, 5),
    inOutSine: t => -(Math.cos(Math.PI * t) - 1) / 2,
    outBack: t => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); },
    smooth: t => t * t * (3 - 2 * t),
  };

  U.clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  U.lerp = (a, b, t) => a + (b - a) * t;
  U.invlerp = (a, b, x) => (x - a) / (b - a);
  U.remap = (x, a, b, c, d) => c + (d - c) * (x - a) / (b - a);
  U.seg = (t, a, b) => U.clamp((t - a) / (b - a), 0, 1);   // local progress of t within [a, b]

  U.css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  U.hexToRgb = hex => {
    const h = hex.replace('#', '');
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  };
  U.rgba = (hex, a) => { const [r, g, b] = U.hexToRgb(hex); return `rgba(${r},${g},${b},${a})`; };
  U.mixHex = (h1, h2, t) => {
    const a = U.hexToRgb(h1), b = U.hexToRgb(h2);
    return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',')})`;
  };

  /* Seeded RNG (mulberry32) and gaussian, as in Every Frame an INR */
  U.mulberry32 = seed => {
    let v = seed >>> 0;
    return () => {
      v += 0x6d2b79f5; let n = v;
      n = Math.imul(n ^ (n >>> 15), n | 1);
      n ^= n + Math.imul(n ^ (n >>> 7), n | 61);
      return ((n ^ (n >>> 14)) >>> 0) / 4294967296;
    };
  };
  U.gaussian = seed => {
    const r = U.mulberry32(seed); let spare = null;
    return () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      const u = Math.max(r(), 1e-7), v = r();
      const rad = Math.sqrt(-2 * Math.log(u)), th = 2 * Math.PI * v;
      spare = rad * Math.sin(th); return rad * Math.cos(th);
    };
  };

  U.fmtInt = n => Math.round(n).toLocaleString('en-US');
  U.fmtCompact = n => {
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1).replace(/\.0$/, '') + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1).replace(/\.0$/, '') + 'k';
    return String(Math.round(n));
  };
  U.fmtSci = (x, d = 1) => {
    if (!isFinite(x)) return '—';
    if (x === 0) return '0';
    const e = Math.floor(Math.log10(Math.abs(x)));
    const m = x / Math.pow(10, e);
    return `${m.toFixed(d)}e${e}`;
  };
  U.psnr = mse => 10 * Math.log10(1 / Math.max(mse, 1e-12));

  /* --- canvas: size a canvas for crisp drawing in design-space px --- */
  U.hidpi = (canvas, w, h) => {
    const pr = (window.Deck ? Deck.pixelRatio() : window.devicePixelRatio || 1);
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    canvas.width = Math.max(1, Math.round(w * pr));
    canvas.height = Math.max(1, Math.round(h * pr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(pr, 0, 0, pr, 0, 0);
    canvas._w = w; canvas._h = h;
    return ctx;
  };

  /* --- svg helpers --- */
  const SVGNS = 'http://www.w3.org/2000/svg';
  U.svg = (tag, attrs = {}, parent = null) => {
    const el = document.createElementNS(SVGNS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(el);
    return el;
  };
  U.el = (tag, attrs = {}, parent = null, html = null) => {
    const el = document.createElement(tag);
    for (const k in attrs) {
      if (k === 'class') el.className = attrs[k];
      else if (k === 'style') el.style.cssText = attrs[k];
      else el.setAttribute(k, attrs[k]);
    }
    if (html !== null) el.innerHTML = html;
    if (parent) parent.appendChild(el);
    return el;
  };

  /* typeset an element (or its children) with KaTeX auto-render */
  U.typeset = el => {
    if (!window.renderMathInElement) return;
    renderMathInElement(el, {
      delimiters: [
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
      ],
      throwOnError: false,
      macros: { '\\R': '\\mathbb{R}' },
    });
  };
  U.tex = (src, el, display = false) => {
    katex.render(src, el, { throwOnError: false, displayMode: display, macros: { '\\R': '\\mathbb{R}' } });
    return el;
  };

  /* load an image from a (data) url */
  U.loadImage = src => new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = src;
  });
  /* image -> Float32Array RGB in [0,1], resized to size x size */
  U.imageToRGB = (img, size) => {
    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, size, size);
    const d = ctx.getImageData(0, 0, size, size).data;
    const out = new Float32Array(size * size * 3);
    for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
      out[j] = d[i] / 255; out[j + 1] = d[i + 1] / 255; out[j + 2] = d[i + 2] / 255;
    }
    return out;
  };
  /* draw an RGB float array (values in [0,1]) into a canvas of the same pixel size */
  U.putRGB = (canvas, rgb, size) => {
    if (canvas.width !== size) { canvas.width = size; canvas.height = size; }
    const ctx = canvas.getContext('2d');
    const im = ctx.createImageData(size, size);
    const d = im.data;
    for (let i = 0, j = 0; j < rgb.length; i += 4, j += 3) {
      d[i] = rgb[j] * 255; d[i + 1] = rgb[j + 1] * 255; d[i + 2] = rgb[j + 2] * 255; d[i + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
  };
  /* grayscale float array -> canvas, with window [lo, hi] */
  U.putGray = (canvas, g, w, h, lo = 0, hi = 1) => {
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext('2d');
    const im = ctx.createImageData(w, h);
    const d = im.data, s = 255 / (hi - lo);
    for (let i = 0, j = 0; j < g.length; i += 4, j++) {
      const v = (g[j] - lo) * s;
      d[i] = d[i + 1] = d[i + 2] = v; d[i + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
  };

  /* simple radix-2 FFT (in place, re/im Float64Arrays, n power of two) */
  U.fft = (re, im) => {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const ar = re[i + k], ai = im[i + k];
          const br = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
          const bi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
          re[i + k] = ar + br; im[i + k] = ai + bi;
          re[i + k + len / 2] = ar - br; im[i + k + len / 2] = ai - bi;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  };

  /* canvas arrow */
  U.arrow = (ctx, x0, y0, x1, y1, head = 14, width = 3, color = '#1d2130') => {
    const a = Math.atan2(y1 - y0, x1 - x0);
    ctx.save();
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - Math.cos(a) * head * 0.8, y1 - Math.sin(a) * head * 0.8); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - head * Math.cos(a - 0.42), y1 - head * Math.sin(a - 0.42));
    ctx.lineTo(x1 - head * Math.cos(a + 0.42), y1 - head * Math.sin(a + 0.42));
    ctx.closePath(); ctx.fill();
    ctx.restore();
  };

  U.roundRect = (ctx, x, y, w, h, r) => {
    r = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  };

  window.U = U;
})();
