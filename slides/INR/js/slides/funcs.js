/* "A function can be much more than a curve": six live examples of functions R^n -> R^m */
(function () {
  const CW = 542, CH = 214;

  /* ---------------- audio: scrolling waveform ---------------- */
  const NOTES = [18, 24, 21, 30, 27, 16, 34, 24, 20, 28, 32, 22];
  function audioSignal(t) {
    const dur = 0.42, n = Math.floor(t / dur), tau = t - n * dur;
    const f = NOTES[((n % NOTES.length) + NOTES.length) % NOTES.length];
    const env = Math.exp(-5.5 * tau) * (1 - Math.exp(-80 * tau));
    const w = 2 * Math.PI * f * t;
    return env * (0.62 * Math.sin(w) + 0.25 * Math.sin(2 * w + 0.4) + 0.12 * Math.sin(3 * w + 1.3));
  }
  function drawAudio(g, t) {
    g.fillStyle = '#f6f5f0'; g.fillRect(0, 0, CW, CH);
    g.strokeStyle = '#dedbd2'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, CH / 2 + 0.5); g.lineTo(CW, CH / 2 + 0.5); g.stroke();
    const win = 1.25, t0 = t * 0.5;
    g.strokeStyle = '#2a78d6'; g.lineWidth = 2.2; g.lineJoin = 'round';
    g.beginPath();
    for (let x = 0; x <= CW; x += 0.5) {
      const v = audioSignal(t0 + x / CW * win);
      const y = CH / 2 - v * CH * 0.42;
      if (x === 0) g.moveTo(x, y); else g.lineTo(x, y);
    }
    g.stroke();
    g.font = 'italic 24px KaTeX_Math, serif'; g.fillStyle = '#8a8780';
    g.fillText('t', CW - 22, CH - 12);
  }

  /* ---------------- image: probe (x,y) -> (r,g,b) ---------------- */
  function drawImageCard(st, g, t) {
    g.fillStyle = '#f6f5f0'; g.fillRect(0, 0, CW, CH);
    if (!st.img) return;
    const S = CH - 24, ox = 14, oy = 12;
    g.save(); U.roundRect(g, ox, oy, S, S, 10); g.clip();
    g.drawImage(st.img, ox, oy, S, S);
    g.restore();
    const u = 0.5 + 0.36 * Math.sin(0.62 * t), v = 0.5 + 0.34 * Math.sin(1.03 * t + 0.6);
    const px = ox + u * S, py = oy + v * S;
    const ix = Math.min(511, Math.max(0, Math.round(u * 511))), iy = Math.min(511, Math.max(0, Math.round(v * 511)));
    const k = (iy * 512 + ix) * 4, d = st.data;
    const r = d[k] / 255, gg = d[k + 1] / 255, b = d[k + 2] / 255;
    // crosshair
    g.strokeStyle = '#fff'; g.lineWidth = 3;
    g.beginPath(); g.arc(px, py, 11, 0, Math.PI * 2); g.stroke();
    g.strokeStyle = 'rgba(0,0,0,0.55)'; g.lineWidth = 1.2;
    g.beginPath(); g.arc(px, py, 13, 0, Math.PI * 2); g.stroke();
    // readout: position in, three channels out
    const tx = ox + S + 28;
    g.textAlign = 'left';
    g.fillStyle = '#1d2130';
    g.font = '25px KaTeX_Main, serif';
    g.fillText(`(x, y) = (${u.toFixed(2)}, ${v.toFixed(2)})`, tx, 46);
    g.font = '30px KaTeX_Main, serif'; g.fillStyle = '#8a8780';
    g.fillText('↦', tx, 86);
    g.fillStyle = `rgb(${d[k]},${d[k + 1]},${d[k + 2]})`;
    U.roundRect(g, tx + 40, 64, 56, 30, 8); g.fill();
    g.strokeStyle = 'rgba(0,0,0,0.14)'; g.lineWidth = 1; g.stroke();
    const ch = [['r', r, '#d64545'], ['g', gg, '#2e9e4a'], ['b', b, '#2a78d6']];
    ch.forEach(([name, val, col], i) => {
      const y = 116 + i * 36, bw = 120;
      g.font = 'italic 25px KaTeX_Math, serif'; g.fillStyle = col;
      g.fillText(name, tx, y + 8);
      g.fillStyle = '#ebe8e0'; U.roundRect(g, tx + 24, y - 9, bw, 18, 5); g.fill();
      g.fillStyle = col; U.roundRect(g, tx + 24, y - 9, Math.max(3, bw * val), 18, 5); g.fill();
      g.font = '22px KaTeX_Main, serif'; g.fillStyle = '#1d2130';
      g.fillText(val.toFixed(2), tx + 24 + bw + 10, y + 8);
    });
  }

  /* ---------------- video: frames stacked along t ---------------- */
  function drawVideo(st, g, t) {
    g.fillStyle = '#f6f5f0'; g.fillRect(0, 0, CW, CH);
    if (!st.frames.length) return;
    const n = st.frames.length, cur = Math.floor(Math.max(0, t) * 3.2) % n;
    const s = 0.72, ax = 0.42 * s, ay = -0.2 * s;
    for (let k = 0; k < n; k++) {
      const ox = 26 + k * 25, oy = 58;
      g.save();
      g.setTransform(1, 0, 0, 1, 0, 0);
      const pr = Deck.pixelRatio();
      g.setTransform(ax * pr, ay * pr, 0, s * pr, ox * pr, oy * pr);
      g.globalAlpha = k === cur ? 1 : 0.42;
      g.drawImage(st.frames[k], 0, 0, 200, 200);
      if (k === cur) { g.strokeStyle = '#2a78d6'; g.lineWidth = 7; g.strokeRect(0, 0, 200, 200); }
      g.restore();
    }
    // t axis
    U.arrow(g, 30, 198, 320, 198, 12, 2.5, '#8a8780');
    g.font = 'italic 22px KaTeX_Math, serif'; g.fillStyle = '#4b4f5c'; g.fillText('t', 326, 206);
    // the current frame, flat
    const fx = 362, fy = 18, fs = 158;
    g.save(); U.roundRect(g, fx, fy, fs, fs, 10); g.clip();
    g.drawImage(st.frames[cur], fx, fy, fs, fs);
    g.restore();
    g.font = '20px "Fira Mono", monospace'; g.fillStyle = '#8a8780';
    g.fillText(`frame ${String(cur + 1).padStart(2, ' ')} / ${n}`, fx + 14, fy + fs + 26);
  }

  /* ---------------- SDF: ray-marched shape + slice of the distance field ---------------- */
  const SDF_FS = `
    uniform vec2 uRes; uniform float uTime;
    mat3 rot(float a, float b){ float ca=cos(a), sa=sin(a), cb=cos(b), sb=sin(b);
      return mat3(ca,0.,-sa, 0.,1.,0., sa,0.,ca) * mat3(1.,0.,0., 0.,cb,sb, 0.,-sb,cb); }
    float smin(float a, float b, float k){ float h=clamp(0.5+0.5*(b-a)/k,0.,1.); return mix(b,a,h)-k*h*(1.-h); }
    float shape(vec3 p){
      vec2 q = vec2(length(p.xz)-0.62, p.y);
      float tor = length(q)-0.17;
      float sph = length(p)-0.40;
      float d = smin(tor, sph, 0.22);
      vec3 pb = p - vec3(0.0, 0.0, 0.0);
      float cut = length(pb - vec3(0.0,0.42,0.0)) - 0.22;
      return max(d, -cut);
    }
    float map(vec3 p, mat3 R){ return shape(R*p); }
    vec3 nrm(vec3 p, mat3 R){ vec2 e=vec2(1e-3,0.); return normalize(vec3(map(p+e.xyy,R)-map(p-e.xyy,R), map(p+e.yxy,R)-map(p-e.yxy,R), map(p+e.yyx,R)-map(p-e.yyx,R))); }
    void main(){
      vec2 fc = gl_FragCoord.xy; float half_ = uRes.x*0.5;
      vec3 bg = vec3(0.965,0.961,0.941);
      mat3 R = rot(uTime*0.55, 0.55 + 0.25*sin(uTime*0.4));
      if (fc.x < half_) {
        vec2 uv = (fc - vec2(half_*0.5, uRes.y*0.5)) / uRes.y;
        vec3 ro = vec3(0.,0.,2.6), rd = normalize(vec3(uv*1.15, -1.6));
        float t=0.; float d; bool hit=false;
        for(int i=0;i<90;i++){ d = map(ro+rd*t, R); if(d<1e-3){hit=true;break;} t+=d; if(t>6.) break; }
        vec3 col = bg;
        if(hit){ vec3 p=ro+rd*t; vec3 n=nrm(p,R); vec3 l=normalize(vec3(0.5,0.8,0.9));
          float dif=max(dot(n,l),0.); float amb=0.6+0.4*n.y; float rim=pow(1.-max(dot(n,-rd),0.),2.0);
          vec3 base=vec3(0.30,0.58,0.92);
          col = base*(0.55*amb+0.65*dif) + 0.22*rim + vec3(1.)*pow(max(dot(reflect(-l,n),-rd),0.),20.)*0.45; }
        gl_FragColor = vec4(col,1.);
      } else {
        vec2 uv = (fc - vec2(half_*1.5, uRes.y*0.5)) / uRes.y * 2.2;
        float d = map(vec3(uv, 0.0), R);
        vec3 inC = vec3(0.16,0.47,0.84), outC = vec3(0.92,0.41,0.20);
        vec3 col = d>0. ? mix(outC, bg, 0.35) : mix(inC, bg, 0.15);
        col *= 1.0 - 0.55*exp(-5.0*abs(d));
        col = mix(col, bg, 0.25 + 0.25*smoothstep(0.0,1.2,abs(d)));
        float band = abs(fract(d*10.0)-0.5);
        col = mix(col*0.86, col, smoothstep(0.0,0.06,band));
        col = mix(vec3(0.11,0.13,0.19), col, smoothstep(0.004,0.016,abs(d)));
        float px = 1.0/uRes.y;
        if (abs(fc.x-half_) < 1.0) col = vec3(0.86,0.85,0.81);
        gl_FragColor = vec4(col,1.);
      }
    }`;

  /* ---------------- radiance field: volumetric rendering with an orbiting camera ---------------- */
  const NERF_FS = `
    uniform vec2 uRes; uniform float uTime;
    float hash(vec3 p){ p=fract(p*0.3183099+.1); p*=17.0; return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }
    float noise(vec3 x){ vec3 i=floor(x), f=fract(x); f=f*f*(3.-2.*f);
      return mix(mix(mix(hash(i),hash(i+vec3(1,0,0)),f.x), mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),
                 mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x), mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y), f.z); }
    float fbm(vec3 p){ return 0.55*noise(p) + 0.3*noise(p*2.1) + 0.15*noise(p*4.3); }
    float sdf(vec3 p){
      vec3 pt = p - vec3(0.0, 0.08, 0.0);
      vec2 q = vec2(length(pt.xy)-0.42, pt.z);          // upright ring
      float ring = length(q) - 0.13;
      float ball = length(p - vec3(0.0,-0.22,0.0)) - 0.26;
      return min(ring, ball);
    }
    float density(vec3 p){
      float d = sdf(p);
      if (d > 0.18) return 0.0;
      float n = fbm(p*5.0);
      return clamp((0.02 - d) * 70.0 + (n - 0.5) * 9.0, 0.0, 40.0);
    }
    vec3 color(vec3 p, vec3 rd){
      vec3 a = vec3(0.10,0.69,0.48), b = vec3(0.16,0.47,0.84), c = vec3(0.92,0.41,0.20);
      float h = clamp(0.55 + p.y*1.3, 0.0, 1.0);
      vec3 base = h < 0.5 ? mix(c,a,h*2.0) : mix(a,b,(h-0.5)*2.0);
      vec3 n = normalize(vec3(sdf(p+vec3(.01,0,0))-sdf(p-vec3(.01,0,0)), sdf(p+vec3(0,.01,0))-sdf(p-vec3(0,.01,0)), sdf(p+vec3(0,0,.01))-sdf(p-vec3(0,0,.01))));
      float dif = 0.55 + 0.45*max(dot(n, normalize(vec3(0.4,0.9,0.3))), 0.0);
      float spec = pow(max(dot(reflect(rd, n), normalize(vec3(0.4,0.9,0.3))), 0.0), 16.0);   // view dependent
      return base*dif + 0.45*spec;
    }
    void main(){
      vec2 uv = (gl_FragCoord.xy - 0.5*uRes) / uRes.y;
      float a = uTime*0.55;
      vec3 ro = vec3(2.2*sin(a), 0.85, 2.2*cos(a));
      vec3 ta = vec3(0.0, -0.1, 0.0);
      vec3 fw = normalize(ta-ro), rt = normalize(cross(fw, vec3(0,1,0))), up = cross(rt, fw);
      vec3 rd = normalize(fw*1.55 + uv.x*rt + uv.y*up);
      vec3 bg = vec3(0.965,0.961,0.941);
      // ground plane y = -0.48 with a grid
      vec3 bgc = bg;
      float tp = (-0.48 - ro.y) / rd.y;
      if (tp > 0.0) {
        vec3 gp = ro + rd*tp;
        vec2 g = abs(fract(gp.xz*3.0) - 0.5);
        float line = 1.0 - smoothstep(0.0, 0.035, min(g.x, g.y));
        float fade = exp(-0.35*dot(gp.xz, gp.xz));
        float sh = smoothstep(0.0, 0.5, length(gp.xz)) * 0.25 + 0.75;
        bgc = mix(bg, mix(vec3(0.90,0.89,0.86), vec3(0.70,0.69,0.66), line) * sh, fade);
      }
      vec3 acc = vec3(0.); float T = 1.0;
      float t = 1.3; float dt = 0.03;
      for (int i=0; i<64; i++){
        vec3 p = ro + rd*t;
        float s = density(p);
        if (s > 0.01) { float al = 1.0 - exp(-s*dt); acc += T*al*color(p, rd); T *= 1.0-al; if (T<0.02) break; }
        t += dt;
        if (tp > 0.0 && t > tp) break;
      }
      gl_FragColor = vec4(acc + T*bgc, 1.0);
    }`;

  /* ---------------- physical field: wind particles over a temperature field ---------------- */
  function flow(x, y, t) {
    // divergence-free field from a stream function psi = sum a sin(kx x + ky y + w t) (+ drift)
    let u = 0.55, v = 0.0;
    const M = [[0.9, 1.6, 2.1, 0.35, 0.4], [-1.4, 2.3, 1.1, 0.42, 1.7], [2.6, -0.9, 3.2, 0.28, 2.9], [1.1, 3.4, 0.6, 0.18, 0.8]];
    for (const [kx, ky, w, a, ph] of M) {
      const c = Math.cos(kx * x + ky * y + 0.5 * w * t + ph) * a;
      u += c * ky; v -= c * kx;
    }
    return [u, v];
  }
  function temp(x, y, t) {
    return 0.5 + 0.28 * Math.sin(1.3 * x - 0.7 * y + 0.3 * t) + 0.18 * Math.sin(0.6 * x + 2.1 * y - 0.25 * t + 1.0) + 0.1 * Math.cos(2.7 * x + 0.4 * t);
  }
  const COLD = [42, 120, 214], MID = [240, 239, 236], WARM = [227, 73, 72];
  function tcol(v) {
    v = U.clamp(v, 0, 1);
    const a = v < 0.5 ? COLD : MID, b = v < 0.5 ? MID : WARM, s = v < 0.5 ? v * 2 : (v - 0.5) * 2;
    return [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s];
  }

  Deck.register('s-funcs', {
    steps: 6,
    init(ctx) {
      const cards = {};
      ctx.$$('.fn-card').forEach(c => { cards[c.dataset.k] = c.querySelector('canvas'); });
      this.cards = cards;
      this.st = { img: null, data: null, frames: [] };
      U.loadImage(ASSETS.media.colorful512).then(im => {
        const c = document.createElement('canvas'); c.width = 512; c.height = 512;
        const g = c.getContext('2d', { willReadFrequently: true });
        g.drawImage(im, 0, 0, 512, 512);
        this.st.data = g.getImageData(0, 0, 512, 512).data;
        this.st.img = im;
      });
      Promise.all(ASSETS.media.video.map(U.loadImage)).then(ims => { this.st.frames = ims; });
      try { this.sdf = new ShaderCanvas(cards.sdf, SDF_FS, { w: CW, h: CH }); } catch (e) { console.error(e); }
      try { this.nerf = new ShaderCanvas(cards.nerf, NERF_FS, { w: CW, h: CH }); } catch (e) { console.error(e); }
      // field: offscreen low-res temperature + trail layer
      this.tlo = document.createElement('canvas'); this.tlo.width = 96; this.tlo.height = 42;
      this.trail = document.createElement('canvas');
      this.parts = Array.from({ length: 700 }, () => ({ x: Math.random() * 6, y: Math.random() * 2.6, age: Math.random() * 120 }));
      this.resize();
    },
    resize() {
      this.g = {
        audio: U.hidpi(this.cards.audio, CW, CH),
        image: U.hidpi(this.cards.image, CW, CH),
        video: U.hidpi(this.cards.video, CW, CH),
        field: U.hidpi(this.cards.field, CW, CH),
      };
      if (this.sdf) this.sdf.resize(CW, CH);
      if (this.nerf) this.nerf.resize(CW, CH);
      const pr = Deck.pixelRatio();
      this.trail.width = Math.round(CW * pr); this.trail.height = Math.round(CH * pr);
      this.tg = this.trail.getContext('2d');
      this.tg.setTransform(pr, 0, 0, pr, 0, 0);
    },
    drawField(t, dt) {
      const g = this.g.field, X = 6, Y = 2.6;
      // temperature (low res, smooth upscale)
      if (!this.tframe || (this.tframe++ % 3 === 0)) {
        this.tframe = this.tframe || 1;
        const lg = this.tlo.getContext('2d'), im = lg.createImageData(96, 42);
        for (let j = 0; j < 42; j++) for (let i = 0; i < 96; i++) {
          const [r, gg, b] = tcol(temp(i / 95 * X, j / 41 * Y, t));
          const k = (j * 96 + i) * 4; im.data[k] = r; im.data[k + 1] = gg; im.data[k + 2] = b; im.data[k + 3] = 255;
        }
        lg.putImageData(im, 0, 0);
      }
      g.imageSmoothingEnabled = true;
      g.globalAlpha = 0.55; g.drawImage(this.tlo, 0, 0, CW, CH); g.globalAlpha = 1;
      g.fillStyle = 'rgba(246,245,240,0.35)'; g.fillRect(0, 0, CW, CH);
      // trails
      const tg = this.tg;
      tg.globalCompositeOperation = 'destination-out';
      tg.fillStyle = 'rgba(0,0,0,0.075)'; tg.fillRect(0, 0, CW, CH);
      tg.globalCompositeOperation = 'source-over';
      tg.strokeStyle = 'rgba(29,33,48,0.62)'; tg.lineWidth = 1.3; tg.lineCap = 'round';
      tg.beginPath();
      const sc = CW / X;
      for (const p of this.parts) {
        const [u, v] = flow(p.x, p.y, t);
        const nx = p.x + u * dt * 0.9, ny = p.y + v * dt * 0.9;
        tg.moveTo(p.x * sc, p.y * sc); tg.lineTo(nx * sc, ny * sc);
        p.x = nx; p.y = ny; p.age++;
        if (p.x < -0.1 || p.x > X + 0.1 || p.y < -0.1 || p.y > Y + 0.1 || p.age > 160 + Math.random() * 80) {
          p.x = Math.random() < 0.5 ? Math.random() * 0.4 : Math.random() * X; p.y = Math.random() * Y; p.age = 0;
        }
      }
      tg.stroke();
      g.drawImage(this.trail, 0, 0, CW, CH);
    },
    enter(ctx) {
      const t0 = performance.now() / 1000;
      ctx.loop((now, dt) => {
        const t = Math.max(0, now - t0);
        drawAudio(this.g.audio, t);
        drawImageCard(this.st, this.g.image, t);
        drawVideo(this.st, this.g.video, t);
        if (this.sdf) this.sdf.draw({ uTime: t });
        if (this.nerf) this.nerf.draw({ uTime: t });
        this.drawField(t, Math.min(dt, 0.05));
      });
    },
  });
})();
