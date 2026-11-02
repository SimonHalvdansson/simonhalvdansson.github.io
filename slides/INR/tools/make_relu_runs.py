"""Train plain ReLU MLPs of increasing size on a 128x128 image and record
snapshots + loss curves for the "Attempt 1: a plain MLP" slide.

Also records a Fourier-feature run of the same image, used as a fallback when
in-browser training (TensorFlow.js) is unavailable.

    python tools/make_relu_runs.py            (~5 min on an M2 Pro with MPS)
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import SOURCES, load_square, to_data_url, write_asset

DEV = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
R = 128
STEPS = 5000
SNAP = sorted(set([0, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 750, 900,
                   1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 4500, 5000]))


def loss_record_steps(n):
    return [s for s in range(1, n + 1) if s <= 100 or s % 5 == 0]


def mlp(inp, width, depth, out=3):
    layers, d = [], inp
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers += [nn.Linear(d, out), nn.Sigmoid()]
    return nn.Sequential(*layers)


class FourierMLP(nn.Module):
    def __init__(self, m, sigma, width, depth):
        super().__init__()
        self.register_buffer('B', torch.randn(2, m) * sigma)
        self.net = mlp(2 * m, width, depth)

    def forward(self, x):
        p = 2 * math.pi * x @ self.B
        return self.net(torch.cat([torch.cos(p), torch.sin(p)], -1))


def train(model, X, y, steps, lr, snaps):
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rec = set(loss_record_steps(steps))
    frames, losses = [], []

    def snapshot(step):
        with torch.no_grad():
            out = model(X).reshape(R, R, 3).clamp(0, 1).cpu().numpy()
            mse = float(((model(X) - y) ** 2).mean())
        frames.append({'step': step, 'mse': round(mse, 7), 'src': to_data_url(out, 'webp', 90)})

    if 0 in snaps:
        snapshot(0)
    t = time.time()
    for s in range(1, steps + 1):
        opt.zero_grad()
        loss = ((model(X) - y) ** 2).mean()
        loss.backward()
        opt.step()
        if s in rec:
            losses.append(round(float(loss), 7))
        if s in snaps:
            snapshot(s)
    n = sum(p.numel() for p in model.parameters())
    print(f'  params={n:,}  final mse={losses[-1]:.2e}  psnr={-10 * math.log10(losses[-1]):.2f}  ({time.time() - t:.0f}s)')
    return n, frames, losses


def main():
    torch.manual_seed(0)
    im = load_square(os.path.join(SOURCES, 'storefront.jpg'), R)
    y = torch.tensor(np.asarray(im) / 255., dtype=torch.float32).reshape(-1, 3).to(DEV)
    xs = torch.linspace(0, 1, R)
    gy, gx = torch.meshgrid(xs, xs, indexing='ij')
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1).to(DEV)

    runs = []
    for (w, d) in [(32, 2), (128, 4), (256, 4), (512, 4)]:
        print(f'ReLU MLP {w}x{d}')
        torch.manual_seed(1)
        n, frames, losses = train(mlp(2, w, d), X, y, STEPS, 3e-3, set(SNAP))
        runs.append({'width': w, 'depth': d, 'params': n, 'frames': frames, 'loss': losses})

    print('Fourier-feature MLP (fallback)')
    torch.manual_seed(1)
    ffsteps = 1000
    n, frames, losses = train(FourierMLP(128, 10, 128, 2), X, y, ffsteps, 3e-3,
                              set(s for s in SNAP if s <= ffsteps))
    ff = {'width': 128, 'depth': 2, 'm': 128, 'sigma': 10, 'params': n, 'frames': frames, 'loss': losses}

    write_asset('relu', {
        'size': R,
        'steps': STEPS,
        'lossSteps': loss_record_steps(STEPS),
        'ffLossSteps': loss_record_steps(ffsteps),
        'target': to_data_url(im, 'png'),
        'runs': runs,
        'ff': ff,
    })


if __name__ == '__main__':
    main()
