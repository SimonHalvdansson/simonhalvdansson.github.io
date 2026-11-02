/* Neural texture compression: a 9-channel material as a fanned stack, a relit render, storage bars */
(function () {
  const RENDER_FS = `
    uniform vec2 uRes; uniform float uTime;
    uniform sampler2D uTex0; uniform sampler2D uTex1; uniform sampler2D uTex2; uniform sampler2D uTex3;
    void main(){
      vec2 uv = gl_FragCoord.xy / uRes;
      // camera looking down at a tilted plane
      vec2 p = (gl_FragCoord.xy - 0.5*uRes) / uRes.y;
      vec3 ro = vec3(0.0, 1.25, 1.35), ta = vec3(0.0, 0.0, -0.05);
      vec3 fw = normalize(ta-ro), rt = normalize(cross(fw, vec3(0,1,0))), up = cross(rt, fw);
      vec3 rd = normalize(fw*1.5 + p.x*rt + p.y*up);
      float t = -ro.y / rd.y;
      vec3 bg = vec3(0.965,0.961,0.941);
      if (t < 0.0) { gl_FragColor = vec4(bg,1.0); return; }
      vec3 pos = ro + rd*t;
      vec2 tc = fract(pos.xz * 0.55 + 0.5);                 // the brick pattern tiles
      vec3 alb = texture2D(uTex0, tc).rgb;
      vec3 nt = texture2D(uTex1, tc).rgb * 2.0 - 1.0;
      float rough = texture2D(uTex2, tc).r;
      float ao = texture2D(uTex3, tc).r;
      vec3 n = normalize(vec3(nt.x, nt.z, nt.y));           // tangent space (x, y up=z) -> world (x, y, z)
      vec3 L = vec3(0.75*cos(uTime*0.8), 0.55, 0.75*sin(uTime*0.8));
      vec3 l = normalize(L - pos);
      vec3 v = normalize(ro - pos);
      vec3 h = normalize(l + v);
      float dif = max(dot(n, l), 0.0);
      float shin = mix(80.0, 6.0, rough);
      float spec = pow(max(dot(n, h), 0.0), shin) * (1.0 - rough) * 0.9;
      float att = 1.6 / (1.0 + 1.2*dot(L-pos, L-pos));
      vec3 col = alb * (0.22*ao + 1.05*dif*att) + vec3(spec*att);
      col = pow(col, vec3(0.92));
      col = mix(col, bg, smoothstep(2.2, 4.2, t));        // distance fog
      gl_FragColor = vec4(col, 1.0);
    }`;

  const BARS = [
    { label: 'Uncompressed (8-bit)', bits: 8, color: '#b9b6ad', note: '' },
    { label: 'Block compression (BC)', bits: 3.94, color: '#eb6834', note: 'PSNR 40.2 dB' },
    { label: 'Neural (NTC)', bits: 1.13, color: '#2a78d6', note: 'PSNR 39.9 dB' },
  ];
  const mb = bits => bits * 4096 * 4096 * 9 / 8 / 1e6;

  Deck.register('s-ntc', {
    steps: 3,
    init(ctx) {
      const stack = this.stackEl = ctx.$('.ntc-stack');
      const mat = ASSETS.media.material;
      this.cards = mat.map(m => {
        const d = U.el('div', { class: 'ntc-card' }, stack);
        U.el('img', { src: m.src, alt: m.label }, d);
        U.el('div', { class: 'lbl' }, d, m.label);
        return d;
      });
      this.rc = ctx.$('#ntc-render');
      this.renderCap = U.el('div', { class: 'ntc-render-cap' }, ctx.$('.ntc-stage'), 'All channels together, lit in real time');
      try {
        this.gl = new ShaderCanvas(this.rc, RENDER_FS, { w: 500, h: 500 });
        const imgs = mat.map(m => U.loadImage(m.src));
        Promise.all(imgs).then(ims => {
          const byKey = Object.fromEntries(mat.map((m, i) => [m.key, ims[i]]));
          [byKey.albedo, byKey.normal, byKey.rough, byKey.ao].forEach((im, i) => this.gl.texture(i, im));
          this.texReady = true;
        });
      } catch (e) { console.error(e); }
      this.bc = ctx.$('#ntc-bars');
      this.resize();
    },
    resize() {
      if (this.gl) this.gl.resize(500, 500);
      this.gb = U.hidpi(this.bc, 760, 300);
      this.drawBars(this.barT ?? 0);
    },
    layout(mode) {
      const n = this.cards.length;
      this.cards.forEach((c, i) => {
        const k = i - (n - 1) / 2;
        if (mode === 'fan') {
          c.style.transform = `translate3d(${k * 118}px, ${-k * 46}px, ${-i * 40}px) rotateY(-30deg) rotateX(12deg) rotateZ(-2deg) scale(0.9)`;
          c.style.opacity = 1;
          c.style.zIndex = n - i;
        } else {
          // fold the channels into the rendered material
          c.style.transform = `translate3d(${k * 8}px, ${30 - k * 8}px, ${-i * 4}px) scale(0.9)`;
          c.style.opacity = 0;
        }
      });
      this.rc.style.opacity = mode === 'fan' ? 0 : 1;
      this.rc.style.transform = mode === 'fan' ? 'scale(0.9)' : 'none';
      this.rc.style.transition = 'opacity 900ms 200ms, transform 1100ms';
      if (this.renderCap) this.renderCap.style.opacity = mode === 'fan' ? 0 : 1;
    },
    drawBars(t) {
      this.barT = t;
      const g = this.gb, Wd = 760;
      g.clearRect(0, 0, Wd, 300);
      const x0 = 0, maxW = 470, max = 8;
      BARS.forEach((b, i) => {
        const y = 20 + i * 92;
        g.font = '500 24px "Fira Sans"'; g.fillStyle = '#1d2130'; g.textAlign = 'left';
        g.fillText(b.label, x0, y + 4);
        const w = maxW * (b.bits / max) * U.Ease.outCubic(U.clamp(t * 1.4 - i * 0.2, 0, 1));
        g.fillStyle = b.color;
        U.roundRect(g, x0, y + 18, Math.max(4, w), 30, 6); g.fill();
        g.font = '400 22px "Fira Mono", monospace'; g.fillStyle = '#4b4f5c';
        const txt = `${b.bits} bits/value · ${Math.round(mb(b.bits))} MB`;
        g.fillText(txt, x0 + Math.max(4, w) + 14, y + 41);
        if (b.note) { g.font = '400 20px "Fira Sans"'; g.fillStyle = '#8a8780'; g.fillText(b.note, x0 + 470 + 14, y + 4); }
      });
    },
    enter(ctx, step) {
      this.layout(step >= 1 ? 'stack' : 'fan');
      this.drawBars(step >= 3 ? 1 : 0);
      const t0 = performance.now() / 1000;
      ctx.loop(now => {
        const t = Math.max(0, now - t0);
        if (this.gl && this.texReady && this.rc.style.opacity !== '0') this.gl.draw({ uTime: t });
        // gentle sway of the fanned stack
        this.stackEl.style.transform = Deck.stepIdx < 1 ? `rotateY(${Math.sin(t * 0.5) * 6}deg) rotateX(${Math.sin(t * 0.37) * 3}deg)` : 'none';
      });
    },
    step(ctx, step, prev) {
      if (step >= 1 && prev < 1) this.layout('stack');
      if (step < 1) this.layout('fan');
      if (step >= 3 && prev < 3) ctx.run(async c => c.play(1300, t => this.drawBars(t), U.Ease.linear));
      if (step < 3) this.drawBars(0);
    },
  });
})();
