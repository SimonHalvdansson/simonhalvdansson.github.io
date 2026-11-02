/* mlpsvg.js — SVG diagram of a small MLP with animated "signal" pulses */
(function () {
  class MLPDiagram {
    /* opts: layers (node counts), x, y, w, h, r, gapLabel (index of layer to draw ⋮ in),
             inLabels, outLabels (arrays of strings, drawn inside nodes), nodeFill, edge */
    constructor(svg, opts) {
      this.svg = svg;
      this.o = Object.assign({ x: 30, y: 20, w: 560, h: 260, r: 15, nodeFill: '#ffffff', stroke: '#1d2130', edge: '#cfccc3',
        edgeWidth: 1.6, layerColors: null, ellipsis: [] }, opts);
      this.g = U.svg('g', {}, svg);
      this.build();
    }
    build() {
      const o = this.o, L = o.layers.length;
      this.g.innerHTML = '';
      this.pos = o.layers.map((n, l) => {
        const x = o.x + (L === 1 ? 0 : (l / (L - 1)) * o.w);
        const pts = [];
        const span = Math.min(o.h, (n - 1) * o.r * 3.2);
        for (let i = 0; i < n; i++) pts.push([x, o.y + o.h / 2 + (n === 1 ? 0 : (i / (n - 1) - 0.5) * span)]);
        return pts;
      });
      // layers listed in o.ellipsis draw "⋮" in place of their middle node
      const gap = this.pos.map((pts, l) => ((o.ellipsis || []).includes(l) && pts.length >= 3 ? Math.floor(pts.length / 2) : -1));
      this.edgeG = U.svg('g', {}, this.g);
      this.pulseG = U.svg('g', {}, this.g);
      this.nodeG = U.svg('g', {}, this.g);
      this.edges = [];
      for (let l = 0; l < L - 1; l++) {
        const arr = [];
        this.pos[l].forEach((a, i) => this.pos[l + 1].forEach((b, j) => {
          if (i === gap[l] || j === gap[l + 1]) return;
          arr.push({ a, b, el: U.svg('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: o.edge, 'stroke-width': o.edgeWidth }, this.edgeG) });
        }));
        this.edges.push(arr);
      }
      this.nodes = this.pos.map((pts, l) => pts.map(([x, y], i) => {
        if (i === gap[l]) {
          U.svg('text', { x, y: y + 10, 'text-anchor': 'middle', 'font-size': 32, fill: o.stroke, 'font-family': 'KaTeX_Main, serif' }, this.nodeG).textContent = '⋮';
          return null;
        }
        return U.svg('circle', { cx: x, cy: y, r: o.r, fill: o.nodeFill, stroke: (o.layerColors && o.layerColors[l]) || o.stroke, 'stroke-width': 2.4 }, this.nodeG);
      }).filter(Boolean));
      this.posVisible = this.pos.map((pts, l) => pts.filter((_, i) => i !== gap[l]));
    }
    /* add a text label inside or beside a node */
    label(l, i, text, opts = {}) {
      const [x, y] = this.posVisible[l][i];
      const t = U.svg('text', {
        x: x + (opts.dx || 0), y: y + (opts.dy || 0) + (opts.size || 22) * 0.34, 'text-anchor': opts.anchor || 'middle',
        'font-size': opts.size || 22, fill: opts.color || '#1d2130',
        'font-family': opts.font || 'KaTeX_Math, KaTeX_Main, serif', 'font-style': opts.italic === false ? 'normal' : 'italic',
      }, this.g);
      t.textContent = text;
      return t;
    }
    /* animate pulses flowing input -> output; returns promise resolving when done */
    pulse(ctx, { dur = 900, color = '#2a78d6', r = 4.5, stagger = 0 } = {}) {
      const L = this.edges.length;
      const dots = this.edges.map(arr => arr.map(() => U.svg('circle', { r, fill: color, opacity: 0 }, this.pulseG)));
      return ctx.play(dur * L, (t, raw) => {
        const T = raw * L;
        for (let l = 0; l < L; l++) {
          const u = U.clamp(T - l, 0, 1);
          const e = U.Ease.inOutSine(u);
          const vis = u > 0 && u < 1;
          this.edges[l].forEach((ed, k) => {
            const d = dots[l][k];
            d.setAttribute('cx', U.lerp(ed.a[0], ed.b[0], e));
            d.setAttribute('cy', U.lerp(ed.a[1], ed.b[1], e));
            d.setAttribute('opacity', vis ? Math.sin(Math.PI * u) * 0.9 : 0);
          });
          this.nodes[l + 1].forEach(n => n.setAttribute('fill', u >= 1 && T - l < 1.6 ? U.mixHex('#ffffff', color, 0.35 * (1 - U.clamp(T - l - 1, 0, 0.6) / 0.6)) : this.o.nodeFill));
        }
      }, U.Ease.linear).finally(() => { dots.flat().forEach(d => d.remove()); this.nodes.flat().forEach(n => n.setAttribute('fill', this.o.nodeFill)); });
    }
  }
  window.MLPDiagram = MLPDiagram;
})();
