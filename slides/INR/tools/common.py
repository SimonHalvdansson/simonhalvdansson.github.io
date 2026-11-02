"""Shared helpers for generating the talk's precomputed assets.

All assets are written as small JavaScript files (assets/*.js) that attach base64
data to window.ASSETS, so the presentation also works when opened via file://.
"""
import base64
import io
import json
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, 'assets')
SOURCES = os.path.join(ROOT, 'tools', 'sources')


def center_square(im):
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def load_square(path, size, crop=None):
    im = Image.open(path).convert('RGB')
    im = im.crop(crop) if crop else center_square(im)
    return im.resize((size, size), Image.LANCZOS)


def to_data_url(arr_or_im, fmt='webp', quality=92):
    im = arr_or_im
    if isinstance(arr_or_im, np.ndarray):
        a = np.clip(arr_or_im, 0, 1)
        im = Image.fromarray((a * 255 + 0.5).astype(np.uint8))
    buf = io.BytesIO()
    if fmt == 'webp':
        im.save(buf, 'WEBP', quality=quality, method=6)
        mime = 'image/webp'
    elif fmt == 'png':
        im.save(buf, 'PNG', optimize=True)
        mime = 'image/png'
    else:
        im.save(buf, 'JPEG', quality=quality)
        mime = 'image/jpeg'
    return f'data:{mime};base64,' + base64.b64encode(buf.getvalue()).decode()


def write_asset(name, obj):
    os.makedirs(ASSETS, exist_ok=True)
    path = os.path.join(ASSETS, f'{name}.js')
    with open(path, 'w') as f:
        f.write('window.ASSETS = window.ASSETS || {};\n')
        f.write(f'window.ASSETS[{json.dumps(name)}] = ')
        json.dump(obj, f, separators=(',', ':'))
        f.write(';\n')
    print(f'wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)')
