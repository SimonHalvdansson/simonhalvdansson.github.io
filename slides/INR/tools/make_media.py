"""Generate assets/media.js: display images, title-slide training frames, video frames
and a procedural 9-channel PBR material (for the neural texture compression slide).

    python tools/make_media.py
"""
import math
import os

import av
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import ndimage

from common import SOURCES, load_square, to_data_url, write_asset

DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'


# ----------------------------------------------------------------------------- title frames
def hero_frames(path, R=320, steps=1500):
    """Fourier-feature MLP learning an image; snapshots at roughly log-spaced steps."""
    torch.manual_seed(2)
    im = load_square(path, R)
    y = torch.tensor(np.asarray(im) / 255., dtype=torch.float32).reshape(-1, 3).to(DEV)
    xs = torch.linspace(0, 1, R)
    gy, gx = torch.meshgrid(xs, xs, indexing='ij')
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1).to(DEV)
    B = (torch.randn(2, 256) * 14).to(DEV)

    net = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3), nn.Sigmoid()).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)

    def f(x):
        p = 2 * math.pi * x @ B
        return net(torch.cat([torch.cos(p), torch.sin(p)], -1))

    snaps = sorted(set([0, 1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 55, 75, 100, 140, 200, 280, 400, 550, 750, 1000, 1500]))
    frames = []
    for s in range(0, steps + 1):
        if s > 0:
            opt.zero_grad()
            loss = ((f(X) - y) ** 2).mean()
            loss.backward()
            opt.step()
        if s in snaps:
            with torch.no_grad():
                out = f(X)
                mse = float(((out - y) ** 2).mean())
                frames.append({'step': s, 'psnr': round(-10 * math.log10(mse), 2),
                               'src': to_data_url(out.reshape(R, R, 3).cpu().numpy(), 'webp', 88)})
    print(f'hero: final psnr {frames[-1]["psnr"]}')
    return {'size': R, 'frames': frames, 'target': to_data_url(im, 'webp', 90)}


# ----------------------------------------------------------------------------- video frames
def video_frames(path, n=10, size=200):
    c = av.open(path)
    frames = [f.to_image() for f in c.decode(video=0)]
    c.close()
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    out = []
    for i in idx:
        im = frames[i].convert('RGB')
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.LANCZOS)
        out.append(to_data_url(im, 'webp', 85))
    print(f'video: {len(frames)} frames -> {n}')
    return out


# ----------------------------------------------------------------------------- material
def fbm(shape, rng, octaves=6, base=4, persistence=0.55):
    h, w = shape
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        g = rng.standard_normal((n + 3, n + 3))
        z = ndimage.zoom(g, ((h + 3 * h / n) / (n + 3), (w + 3 * w / n) / (n + 3)), order=3)[:h, :w]
        out += amp * z
        tot += amp
        amp *= persistence
    out /= tot
    return (out - out.mean()) / (out.std() + 1e-9)


def material(S=512, seed=4):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    bh, bw, mortar = S / 8, S / 4, 7.0
    row = np.floor(yy / bh)
    off = (row % 2) * bw / 2
    col = np.floor((xx + off) / bw)
    lx = (xx + off) - col * bw
    ly = yy - row * bh
    d = np.minimum(np.minimum(lx, bw - lx), np.minimum(ly, bh - ly))      # distance to brick edge
    brick_id = (row * 37 + col * 11).astype(int)
    ids = np.unique(brick_id)
    tint = {i: rng.uniform(-1, 1, 3) for i in ids}
    tint_img = np.stack([np.vectorize(lambda i, k=k: tint[i][k])(brick_id) for k in range(3)], -1)

    n1 = fbm((S, S), rng, 6, 4)
    n2 = fbm((S, S), rng, 5, 16)
    chips = (fbm((S, S), rng, 4, 8) > 1.6).astype(float)

    bevel = np.clip((d - mortar / 2) / 6.0, 0, 1)
    bevel = bevel * bevel * (3 - 2 * bevel)
    height = 0.22 + 0.78 * bevel
    height = height - 0.05 * n2 * bevel - 0.25 * chips * bevel + 0.04 * n1
    height = ndimage.gaussian_filter(height, 0.8)
    height = (height - height.min()) / (height.max() - height.min())

    is_brick = bevel > 0.5
    base_brick = np.array([0.62, 0.30, 0.20])
    base_mortar = np.array([0.72, 0.70, 0.66])
    alb = np.where(is_brick[..., None], base_brick + 0.07 * tint_img, base_mortar)
    alb = alb * (1 + 0.10 * n2[..., None]) * (1 + 0.06 * n1[..., None])
    soot = np.clip(fbm((S, S), rng, 5, 3) * 0.5, 0, 1)[..., None]
    alb = alb * (1 - 0.35 * soot)
    alb = np.clip(alb, 0, 1)

    gy_, gx_ = np.gradient(height)
    k = 9.0
    nrm = np.stack([-gx_ * k, -gy_ * k, np.ones_like(height)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)

    rough = np.clip(np.where(is_brick, 0.62, 0.92) + 0.10 * n2 - 0.15 * soot[..., 0], 0, 1)
    blur = ndimage.gaussian_filter(height, 6)
    ao = np.clip(1 - 2.2 * np.clip(blur - height, 0, None), 0, 1) ** 1.5
    ao = 0.25 + 0.75 * ao

    gray = lambda a: np.repeat(a[..., None], 3, -1)
    maps = [
        ('albedo', 'Albedo · 3', alb),
        ('normal', 'Normal · 3', nrm * 0.5 + 0.5),
        ('rough', 'Roughness · 1', gray(rough)),
        ('ao', 'Ambient occl. · 1', gray(ao)),
        ('height', 'Height · 1', gray(height)),
    ]
    return [{'key': k_, 'label': lbl, 'src': to_data_url(a, 'webp', 90)} for k_, lbl, a in maps]


def main():
    sf512 = load_square(os.path.join(SOURCES, 'storefront.jpg'), 512)
    sf128 = load_square(os.path.join(SOURCES, 'storefront.jpg'), 128)
    write_asset('media', {
        'storefront512': to_data_url(sf512, 'webp', 90),
        'storefront128': to_data_url(sf128, 'png'),
        'colorful512': to_data_url(load_square(os.path.join(SOURCES, 'color-folds.jpg'), 512), 'webp', 90),
        'hero': hero_frames(os.path.join(SOURCES, 'coastal-rocks.jpg')),
        'video': video_frames(os.path.join(SOURCES, 'city-tram.mp4')),
        'material': material(),
    })


if __name__ == '__main__':
    main()
