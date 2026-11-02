/* gl.js — minimal WebGL helper: a full-screen fragment shader with uniforms */
(function () {
  const VS = `attribute vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }`;

  class ShaderCanvas {
    constructor(canvas, fragSrc, { w, h } = {}) {
      this.canvas = canvas;
      this.w = w; this.h = h;
      const gl = this.gl = canvas.getContext('webgl', { antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: false });
      if (!gl) { this.failed = true; return; }
      const sh = (type, src) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        return s;
      };
      const prog = this.prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
      gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, 'precision highp float;\n' + fragSrc));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(prog, 'p');
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      this.uni = {};
      this.textures = [];
      this.resize();
    }
    resize(w = this.w, h = this.h) {
      if (this.failed) return;
      this.w = w; this.h = h;
      const pr = Math.min(2.5, window.Deck ? Deck.pixelRatio() : window.devicePixelRatio || 1);
      this.canvas.style.width = w + 'px';
      this.canvas.style.height = h + 'px';
      this.canvas.width = Math.round(w * pr);
      this.canvas.height = Math.round(h * pr);
    }
    u(name) {
      if (!(name in this.uni)) this.uni[name] = this.gl.getUniformLocation(this.prog, name);
      return this.uni[name];
    }
    texture(unit, img) {
      const gl = this.gl;
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.textures[unit] = t;
      return t;
    }
    draw(uniforms = {}) {
      if (this.failed) return;
      const gl = this.gl;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.useProgram(this.prog);
      gl.uniform2f(this.u('uRes'), this.canvas.width, this.canvas.height);
      for (const k in uniforms) {
        const v = uniforms[k], loc = this.u(k);
        if (loc === null) continue;
        if (typeof v === 'number') gl.uniform1f(loc, v);
        else if (v.length === 2) gl.uniform2fv(loc, v);
        else if (v.length === 3) gl.uniform3fv(loc, v);
        else if (v.length === 4) gl.uniform4fv(loc, v);
      }
      this.textures.forEach((t, i) => {
        if (!t) return;
        gl.activeTexture(gl.TEXTURE0 + i);
        gl.bindTexture(gl.TEXTURE_2D, t);
        const loc = this.u('uTex' + i);
        if (loc !== null) gl.uniform1i(loc, i);
      });
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
  }
  window.ShaderCanvas = ShaderCanvas;
})();
